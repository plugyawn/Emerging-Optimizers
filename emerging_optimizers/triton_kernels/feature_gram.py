# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Triton kernels for FEATURE_GRAM diagonal paths.

The public helpers always provide a PyTorch fallback. CUDA/Triton users get a
single-kernel diagonal Gram reduction and a single-kernel diagonal
right-preconditioned parameter update.
"""

from __future__ import annotations

import torch


try:
    import triton
    import triton.language as tl

    HAS_TRITON_FEATURE_GRAM = True
except ImportError:  # pragma: no cover - depends on optional Triton install.
    triton = None
    tl = None
    HAS_TRITON_FEATURE_GRAM = False


__all__ = [
    "HAS_TRITON_FEATURE_GRAM",
    "apply_diag_right_preconditioned_update_kernel_",
    "diag_feature_gram_reduce",
]


if HAS_TRITON_FEATURE_GRAM:

    @triton.jit
    def _diag_feature_gram_kernel(
        x,
        out,
        M: tl.constexpr,
        N: tl.constexpr,
        stride_m: tl.constexpr,
        stride_n: tl.constexpr,
        ridge: tl.constexpr,
        mean: tl.constexpr,
        reciprocal: tl.constexpr,
        accumulate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for start_m in tl.range(0, M, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            vals = tl.load(
                x + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(vals * vals, axis=0)
        if mean:
            acc = acc / M
        if ridge != 0.0:
            acc = acc + ridge
        if reciprocal:
            acc = 1.0 / acc
        if accumulate:
            acc += tl.load(out + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        tl.store(out + offs_n, acc, mask=offs_n < N)

    @triton.jit
    def _diag_right_update_kernel(
        param,
        grad,
        diag,
        rows: tl.constexpr,
        cols: tl.constexpr,
        p_stride_m: tl.constexpr,
        p_stride_n: tl.constexpr,
        g_stride_m: tl.constexpr,
        g_stride_n: tl.constexpr,
        lr: tl.constexpr,
        ridge: tl.constexpr,
        update_scale: tl.constexpr,
        weight_decay: tl.constexpr,
        decoupled_weight_decay: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
        p = tl.load(
            param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        g = tl.load(
            grad + offs_m[:, None] * g_stride_m + offs_n[None, :] * g_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if weight_decay != 0.0:
            if decoupled_weight_decay:
                p = p * (1.0 - lr * weight_decay)
            else:
                g = g + weight_decay * p
        denom = tl.load(diag + offs_n, mask=offs_n < cols, other=1.0).to(tl.float32) + ridge
        p = p - lr * update_scale * g / denom[None, :]
        tl.store(
            param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
            p,
            mask=mask,
        )


def _diag_feature_gram_fallback(
    x: torch.Tensor,
    *,
    mean: bool,
    ridge: float,
    reciprocal: bool,
) -> torch.Tensor:
    value = torch.sum(x.to(torch.float32) * x.to(torch.float32), dim=0)
    if mean:
        value = value / x.shape[0]
    if ridge != 0.0:
        value = value + ridge
    if reciprocal:
        value = value.reciprocal()
    return value


def diag_feature_gram_reduce(
    x: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
    mean: bool = False,
    ridge: float = 0.0,
    reciprocal: bool = False,
    accumulate: bool = False,
) -> torch.Tensor:
    """Compute diagonal ``X.T @ X`` with optional mean/ridge/reciprocal transforms."""

    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    if out is None:
        out = torch.empty((x.shape[-1],), device=x.device, dtype=torch.float32)
        accumulate = False
    if count is not None:
        count.fill_(float(x.shape[0]))
    if not HAS_TRITON_FEATURE_GRAM or not x.is_cuda:
        value = _diag_feature_gram_fallback(
            x, mean=mean, ridge=ridge, reciprocal=reciprocal
        ).to(out.dtype)
        if accumulate:
            out.add_(value)
        else:
            out.copy_(value)
        return out

    if not x.is_contiguous():
        x = x.contiguous()
    block_m = 128
    block_n = 256
    grid = ((x.shape[-1] + block_n - 1) // block_n,)
    _diag_feature_gram_kernel[grid](
        x,
        out,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        x.stride(1),
        float(ridge),
        bool(mean),
        bool(reciprocal),
        bool(accumulate),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def apply_diag_right_preconditioned_update_kernel_(
    param: torch.Tensor,
    grad: torch.Tensor,
    diag_feature_gram: torch.Tensor,
    *,
    lr: float,
    ridge: float = 0.0,
    update_scale: float = 1.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Fused diagonal right-preconditioned parameter update with fallback."""

    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError("param and grad must be two-dimensional")
    if diag_feature_gram.ndim != 1:
        raise ValueError("diag_feature_gram must be one-dimensional")
    if diag_feature_gram.shape[0] != param.shape[-1]:
        raise ValueError("diag_feature_gram length must match the parameter feature dimension")
    if not HAS_TRITON_FEATURE_GRAM or not (param.is_cuda and grad.is_cuda and diag_feature_gram.is_cuda):
        if weight_decay != 0.0 and decoupled_weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        update_grad = grad
        if weight_decay != 0.0 and not decoupled_weight_decay:
            update_grad = grad.add(param, alpha=weight_decay)
        denom = diag_feature_gram.to(update_grad.dtype) + ridge
        if torch.any(denom <= 0):
            raise ValueError(
                "Diagonal feature_gram entries must be positive after ridge regularization."
            )
        param.add_(update_grad / denom, alpha=-lr * update_scale)
        return param

    block_m = 16
    block_n = 64
    grid = (
        (param.shape[0] + block_m - 1) // block_m,
        (param.shape[1] + block_n - 1) // block_n,
    )
    _diag_right_update_kernel[grid](
        param,
        grad,
        diag_feature_gram,
        param.shape[0],
        param.shape[1],
        param.stride(0),
        param.stride(1),
        grad.stride(0),
        grad.stride(1),
        float(lr),
        float(ridge),
        float(update_scale),
        float(weight_decay),
        bool(decoupled_weight_decay),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return param
