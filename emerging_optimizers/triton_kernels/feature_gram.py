# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Triton kernels for diagonal matrix preconditioner paths.

The public helpers always provide a PyTorch fallback. CUDA/Triton users get a
single-kernel diagonal Gram reduction and single-kernel diagonal left/right
preconditioned parameter updates.

This module deliberately owns diagonal hot paths only. Dense and block-diagonal
FEATURE_GRAM/GRAD_GRAM right/left solves remain in matrix_update_rules.py as
factorized torch.linalg/cuBLAS-backed paths so callers do not mistake this file
for generic Gram-solve acceleration.
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
    "apply_diag_left_preconditioned_update_kernel_",
    "apply_diag_right_preconditioned_update_kernel_",
    "apply_diag_two_sided_preconditioned_update_kernel_",
    "apply_matrix_update_kernel_",
    "diag_grad_gram_reduce",
    "diag_left_precondition_matrix",
    "diag_right_precondition_matrix",
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

    @triton.jit
    def _diag_left_update_kernel(
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
        denom = tl.load(diag + offs_m, mask=offs_m < rows, other=1.0).to(tl.float32) + ridge
        p = p - lr * update_scale * g / denom[:, None]
        tl.store(
            param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
            p,
            mask=mask,
        )

    @triton.jit
    def _diag_two_sided_update_kernel(
        param,
        grad,
        diag_left,
        diag_right,
        rows: tl.constexpr,
        cols: tl.constexpr,
        p_stride_m: tl.constexpr,
        p_stride_n: tl.constexpr,
        g_stride_m: tl.constexpr,
        g_stride_n: tl.constexpr,
        lr: tl.constexpr,
        ridge_left: tl.constexpr,
        ridge_right: tl.constexpr,
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
        denom_left = (
            tl.load(diag_left + offs_m, mask=offs_m < rows, other=1.0).to(tl.float32)
            + ridge_left
        )
        denom_right = (
            tl.load(diag_right + offs_n, mask=offs_n < cols, other=1.0).to(tl.float32)
            + ridge_right
        )
        p = p - lr * update_scale * g / (denom_left[:, None] * denom_right[None, :])
        tl.store(
            param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
            p,
            mask=mask,
        )

    @triton.jit
    def _diag_right_precondition_kernel(
        param,
        grad,
        diag,
        out,
        rows: tl.constexpr,
        cols: tl.constexpr,
        p_stride_m: tl.constexpr,
        p_stride_n: tl.constexpr,
        g_stride_m: tl.constexpr,
        g_stride_n: tl.constexpr,
        o_stride_m: tl.constexpr,
        o_stride_n: tl.constexpr,
        ridge: tl.constexpr,
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
        g = tl.load(
            grad + offs_m[:, None] * g_stride_m + offs_n[None, :] * g_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if weight_decay != 0.0 and not decoupled_weight_decay:
            p = tl.load(
                param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            g = g + weight_decay * p
        denom = tl.load(diag + offs_n, mask=offs_n < cols, other=1.0).to(tl.float32) + ridge
        tl.store(
            out + offs_m[:, None] * o_stride_m + offs_n[None, :] * o_stride_n,
            g / denom[None, :],
            mask=mask,
        )

    @triton.jit
    def _diag_left_precondition_kernel(
        param,
        grad,
        diag,
        out,
        rows: tl.constexpr,
        cols: tl.constexpr,
        p_stride_m: tl.constexpr,
        p_stride_n: tl.constexpr,
        g_stride_m: tl.constexpr,
        g_stride_n: tl.constexpr,
        o_stride_m: tl.constexpr,
        o_stride_n: tl.constexpr,
        ridge: tl.constexpr,
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
        g = tl.load(
            grad + offs_m[:, None] * g_stride_m + offs_n[None, :] * g_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if weight_decay != 0.0 and not decoupled_weight_decay:
            p = tl.load(
                param + offs_m[:, None] * p_stride_m + offs_n[None, :] * p_stride_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            g = g + weight_decay * p
        denom = tl.load(diag + offs_m, mask=offs_m < rows, other=1.0).to(tl.float32) + ridge
        tl.store(
            out + offs_m[:, None] * o_stride_m + offs_n[None, :] * o_stride_n,
            g / denom[:, None],
            mask=mask,
        )

    @triton.jit
    def _matrix_update_kernel(
        param,
        update,
        rows: tl.constexpr,
        cols: tl.constexpr,
        p_stride_m: tl.constexpr,
        p_stride_n: tl.constexpr,
        u_stride_m: tl.constexpr,
        u_stride_n: tl.constexpr,
        lr: tl.constexpr,
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
        u = tl.load(
            update + offs_m[:, None] * u_stride_m + offs_n[None, :] * u_stride_n,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if weight_decay != 0.0 and decoupled_weight_decay:
            p = p * (1.0 - lr * weight_decay)
        p = p - lr * update_scale * u
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


def diag_grad_gram_reduce(
    dy: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    count: torch.Tensor | None = None,
    mean: bool = False,
    ridge: float = 0.0,
    reciprocal: bool = False,
    accumulate: bool = False,
) -> torch.Tensor:
    """Compute diagonal ``dY.T @ dY`` with the same storage contract as FEATURE_GRAM.

    This is intentionally a named wrapper around the diagonal Gram reducer:
    token-major activations ``X`` and output gradients ``dY`` both reduce over
    the token/sample dimension, but call sites should preserve whether the
    result is input-side/right or output-side/left preconditioning metadata.
    """

    return diag_feature_gram_reduce(
        dy,
        out=out,
        count=count,
        mean=mean,
        ridge=ridge,
        reciprocal=reciprocal,
        accumulate=accumulate,
    )


def diag_right_precondition_matrix(
    grad: torch.Tensor,
    diag_feature_gram: torch.Tensor,
    *,
    param: torch.Tensor | None = None,
    ridge: float = 0.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Return ``grad @ diag(diag_feature_gram + ridge)^-1`` as fp32.

    If ``decoupled_weight_decay`` is false, the coupled L2 term is included
    before preconditioning, matching the Megatron MatrixFunctionOptimizer
    convention for generic update rules.
    """

    if grad.ndim != 2:
        raise ValueError("grad must be two-dimensional")
    if diag_feature_gram.ndim != 1:
        raise ValueError("diag_feature_gram must be one-dimensional")
    if diag_feature_gram.shape[0] != grad.shape[-1]:
        raise ValueError("diag_feature_gram length must match the gradient feature dimension")
    if weight_decay != 0.0 and not decoupled_weight_decay and param is None:
        raise ValueError("param is required for coupled weight decay")

    out = torch.empty(grad.shape, device=grad.device, dtype=torch.float32)
    if not HAS_TRITON_FEATURE_GRAM or not (grad.is_cuda and diag_feature_gram.is_cuda):
        update_grad = grad.to(torch.float32)
        if weight_decay != 0.0 and not decoupled_weight_decay:
            update_grad = update_grad + param.to(torch.float32) * weight_decay
        denom = diag_feature_gram.to(update_grad.dtype) + ridge
        if torch.any(denom <= 0):
            raise ValueError(
                "Diagonal feature_gram entries must be positive after ridge regularization."
            )
        out.copy_(update_grad / denom)
        return out

    if weight_decay != 0.0 and not decoupled_weight_decay and not param.is_cuda:
        raise ValueError("param must be CUDA when coupled weight decay is used on CUDA grad")
    if param is None:
        param = grad

    block_m = 16
    block_n = 64
    grid = (
        (grad.shape[0] + block_m - 1) // block_m,
        (grad.shape[1] + block_n - 1) // block_n,
    )
    _diag_right_precondition_kernel[grid](
        param,
        grad,
        diag_feature_gram,
        out,
        grad.shape[0],
        grad.shape[1],
        param.stride(0),
        param.stride(1),
        grad.stride(0),
        grad.stride(1),
        out.stride(0),
        out.stride(1),
        float(ridge),
        float(weight_decay),
        bool(decoupled_weight_decay),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def diag_left_precondition_matrix(
    grad: torch.Tensor,
    diag_grad_gram: torch.Tensor,
    *,
    param: torch.Tensor | None = None,
    ridge: float = 0.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Return ``diag(diag_grad_gram + ridge)^-1 @ grad`` as fp32.

    If ``decoupled_weight_decay`` is false, the coupled L2 term is included
    before preconditioning, matching the Megatron MatrixFunctionOptimizer
    convention for generic update rules.
    """

    if grad.ndim != 2:
        raise ValueError("grad must be two-dimensional")
    if diag_grad_gram.ndim != 1:
        raise ValueError("diag_grad_gram must be one-dimensional")
    if diag_grad_gram.shape[0] != grad.shape[-2]:
        raise ValueError("diag_grad_gram length must match the gradient output dimension")
    if weight_decay != 0.0 and not decoupled_weight_decay and param is None:
        raise ValueError("param is required for coupled weight decay")

    out = torch.empty(grad.shape, device=grad.device, dtype=torch.float32)
    if not HAS_TRITON_FEATURE_GRAM or not (grad.is_cuda and diag_grad_gram.is_cuda):
        update_grad = grad.to(torch.float32)
        if weight_decay != 0.0 and not decoupled_weight_decay:
            update_grad = update_grad + param.to(torch.float32) * weight_decay
        denom = diag_grad_gram.to(update_grad.dtype) + ridge
        if torch.any(denom <= 0):
            raise ValueError(
                "Diagonal grad_gram entries must be positive after ridge regularization."
            )
        out.copy_(update_grad / denom[:, None])
        return out

    if weight_decay != 0.0 and not decoupled_weight_decay and not param.is_cuda:
        raise ValueError("param must be CUDA when coupled weight decay is used on CUDA grad")
    if param is None:
        param = grad

    block_m = 16
    block_n = 64
    grid = (
        (grad.shape[0] + block_m - 1) // block_m,
        (grad.shape[1] + block_n - 1) // block_n,
    )
    _diag_left_precondition_kernel[grid](
        param,
        grad,
        diag_grad_gram,
        out,
        grad.shape[0],
        grad.shape[1],
        param.stride(0),
        param.stride(1),
        grad.stride(0),
        grad.stride(1),
        out.stride(0),
        out.stride(1),
        float(ridge),
        float(weight_decay),
        bool(decoupled_weight_decay),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def apply_matrix_update_kernel_(
    param: torch.Tensor,
    update: torch.Tensor,
    *,
    lr: float,
    update_scale: float = 1.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Apply ``param -= lr * update_scale * update`` with optional decoupled WD."""

    if param.ndim != 2 or update.ndim != 2:
        raise ValueError("param and update must be two-dimensional")
    if param.shape != update.shape:
        raise ValueError("param and update must have matching shapes")
    if not HAS_TRITON_FEATURE_GRAM or not (param.is_cuda and update.is_cuda):
        if weight_decay != 0.0 and decoupled_weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        param.add_(update.to(param.dtype), alpha=-lr * update_scale)
        return param

    block_m = 16
    block_n = 64
    grid = (
        (param.shape[0] + block_m - 1) // block_m,
        (param.shape[1] + block_n - 1) // block_n,
    )
    _matrix_update_kernel[grid](
        param,
        update,
        param.shape[0],
        param.shape[1],
        param.stride(0),
        param.stride(1),
        update.stride(0),
        update.stride(1),
        float(lr),
        float(update_scale),
        float(weight_decay),
        bool(decoupled_weight_decay),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return param


def apply_diag_left_preconditioned_update_kernel_(
    param: torch.Tensor,
    grad: torch.Tensor,
    diag_grad_gram: torch.Tensor,
    *,
    lr: float,
    ridge: float = 0.0,
    update_scale: float = 1.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Fused diagonal left-preconditioned parameter update with fallback."""

    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError("param and grad must be two-dimensional")
    if diag_grad_gram.ndim != 1:
        raise ValueError("diag_grad_gram must be one-dimensional")
    if diag_grad_gram.shape[0] != param.shape[-2]:
        raise ValueError("diag_grad_gram length must match the parameter output dimension")
    if not HAS_TRITON_FEATURE_GRAM or not (param.is_cuda and grad.is_cuda and diag_grad_gram.is_cuda):
        if weight_decay != 0.0 and decoupled_weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        update_grad = grad
        if weight_decay != 0.0 and not decoupled_weight_decay:
            update_grad = grad.add(param, alpha=weight_decay)
        denom = diag_grad_gram.to(update_grad.dtype) + ridge
        if torch.any(denom <= 0):
            raise ValueError(
                "Diagonal grad_gram entries must be positive after ridge regularization."
            )
        param.add_(update_grad / denom[:, None], alpha=-lr * update_scale)
        return param

    block_m = 16
    block_n = 64
    grid = (
        (param.shape[0] + block_m - 1) // block_m,
        (param.shape[1] + block_n - 1) // block_n,
    )
    _diag_left_update_kernel[grid](
        param,
        grad,
        diag_grad_gram,
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


def apply_diag_two_sided_preconditioned_update_kernel_(
    param: torch.Tensor,
    grad: torch.Tensor,
    diag_left: torch.Tensor,
    diag_right: torch.Tensor,
    *,
    lr: float,
    ridge_left: float = 0.0,
    ridge_right: float = 0.0,
    update_scale: float = 1.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
) -> torch.Tensor:
    """Fused diagonal left-and-right-preconditioned parameter update.

    Applies ``param -= lr * update_scale * D_left^-1 @ grad @ D_right^-1``
    without materializing either preconditioned intermediate.
    """

    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError("param and grad must be two-dimensional")
    if param.shape != grad.shape:
        raise ValueError("param and grad must have matching shapes")
    if diag_left.ndim != 1 or diag_right.ndim != 1:
        raise ValueError("diag_left and diag_right must be one-dimensional")
    if diag_left.shape[0] != param.shape[-2]:
        raise ValueError("diag_left length must match the parameter output dimension")
    if diag_right.shape[0] != param.shape[-1]:
        raise ValueError("diag_right length must match the parameter feature dimension")
    if not (
        HAS_TRITON_FEATURE_GRAM
        and param.is_cuda
        and grad.is_cuda
        and diag_left.is_cuda
        and diag_right.is_cuda
    ):
        if weight_decay != 0.0 and decoupled_weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        update_grad = grad
        if weight_decay != 0.0 and not decoupled_weight_decay:
            update_grad = grad.add(param, alpha=weight_decay)
        denom_left = diag_left.to(update_grad.dtype) + ridge_left
        denom_right = diag_right.to(update_grad.dtype) + ridge_right
        if torch.any(denom_left <= 0) or torch.any(denom_right <= 0):
            raise ValueError(
                "Diagonal preconditioner entries must be positive after ridge regularization."
            )
        update = update_grad / (denom_left[:, None] * denom_right[None, :])
        param.add_(update, alpha=-lr * update_scale)
        return param

    block_m = 16
    block_n = 64
    grid = (
        (param.shape[0] + block_m - 1) // block_m,
        (param.shape[1] + block_n - 1) // block_n,
    )
    _diag_two_sided_update_kernel[grid](
        param,
        grad,
        diag_left,
        diag_right,
        param.shape[0],
        param.shape[1],
        param.stride(0),
        param.stride(1),
        grad.stride(0),
        grad.stride(1),
        float(lr),
        float(ridge_left),
        float(ridge_right),
        float(update_scale),
        float(weight_decay),
        bool(decoupled_weight_decay),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return param
