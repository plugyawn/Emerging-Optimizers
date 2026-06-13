# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix update rules that consume affine wgrad factors.

The public functions in this module consume only the ordinary affine weight
gradient ``G = dY.T @ X`` plus optional input-side feature Gram ``C = X.T @ X``
and/or output-side grad Gram ``R = dY.T @ dY``. They do not require raw
activations, LocoProp targets, output-feature crosses, or the current logical
weight as an operand for the update direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
    newton_schulz as _shared_newton_schulz,
)
from emerging_optimizers.utils import fp32_matmul_precision

CoeffIterMode = Literal["cycle", "repeat_last"]
NSCoeffT = Literal["simple", "quintic", "polar_express", "cans", "aol", "custom"]
MuonScaleT = Literal["none", "shape_scaling", "spectral", "unit_rms_norm"]

__all__ = [
    "CoeffIterMode",
    "NSCoeffT",
    "MuonScaleT",
    "apply_diag_newton_muon_update_",
    "apply_diag_left_preconditioned_update_",
    "apply_diag_right_preconditioned_update_",
    "apply_diag_two_sided_preconditioned_update_",
    "block_diag_feature_gram_to_dense",
    "dense_feature_gram_to_block_diag",
    "diag_feature_gram_to_block_diag",
    "factorize_feature_gram",
    "FeatureGramFactorization",
    "feature_gram_to_diag",
    "locoprop_s_update",
    "newton_muon_update",
    "newton_schulz_orthogonalize_grouped",
    "newton_schulz_orthogonalize",
    "right_precondition_with_factorized_feature_gram",
    "right_precondition_with_feature_gram",
]

def newton_schulz_orthogonalize(
    x: torch.Tensor,
    *,
    steps: int,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
    transpose: bool | None = None,
    use_syrk: bool = False,
) -> torch.Tensor:
    """Newton-Schulz/Polar Express wrapper shared with the Muon optimizer."""

    return _shared_newton_schulz(
        x,
        steps=steps,
        coefficient_type=coefficient_type,
        custom_coefficient_sets=custom_coefficient_sets,
        eps=eps,
        transpose=transpose,
        use_syrk=use_syrk,
    )


def _matrix_solve_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _add_ridge_to_dense_or_blocks(feature_gram: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge == 0.0:
        return feature_gram
    eye = torch.eye(
        feature_gram.shape[-1], device=feature_gram.device, dtype=feature_gram.dtype
    )
    return feature_gram + ridge * eye


def _regularize_feature_gram(feature_gram: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge == 0.0:
        return feature_gram
    feature_gram = feature_gram.to(_matrix_solve_dtype(feature_gram.dtype))
    if feature_gram.ndim == 1:
        return feature_gram + ridge
    if feature_gram.ndim in (2, 3):
        return _add_ridge_to_dense_or_blocks(feature_gram, ridge)
    raise ValueError(
        "feature_gram must be diagonal [p], dense [p, p], or block-diagonal [num_blocks, b, b]"
    )


def feature_gram_to_diag(
    feature_gram: torch.Tensor,
    *,
    feature_dim: int | None = None,
) -> torch.Tensor:
    """Project a supported FEATURE_GRAM representation to diagonal storage.

    ``diag`` is lossless for one-dimensional inputs and lossy for dense or
    block-diagonal inputs because off-diagonal correlations are dropped.
    ``feature_dim`` crops padded block-diagonal storage back to the logical
    feature dimension.
    """

    if feature_gram.ndim == 1:
        diag = feature_gram
    elif feature_gram.ndim == 2:
        if feature_gram.shape[-1] != feature_gram.shape[-2]:
            raise ValueError("dense feature_gram must be square")
        diag = torch.diagonal(feature_gram)
    elif feature_gram.ndim == 3:
        if feature_gram.shape[-1] != feature_gram.shape[-2]:
            raise ValueError("block-diagonal feature_gram must have square blocks")
        diag = torch.diagonal(feature_gram, dim1=-2, dim2=-1).reshape(-1)
    else:
        raise ValueError("feature_gram must be diagonal, dense, or block-diagonal")
    if feature_dim is not None:
        return diag[..., :feature_dim]
    return diag


def diag_feature_gram_to_block_diag(
    diag_feature_gram: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Embed a diagonal FEATURE_GRAM into padded block-diagonal storage.

    This conversion is lossless with respect to the diagonal approximation: it
    does not invent missing correlations, it only changes storage layout.
    """

    if diag_feature_gram.ndim != 1:
        raise ValueError("diag_feature_gram must be one-dimensional")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    padded, _ = _pad_feature_axis(diag_feature_gram, block_size)
    num_blocks = padded.numel() // block_size
    blocks = torch.zeros(
        (num_blocks, block_size, block_size),
        device=diag_feature_gram.device,
        dtype=diag_feature_gram.dtype,
    )
    block_diags = padded.reshape(num_blocks, block_size)
    idx = torch.arange(block_size, device=diag_feature_gram.device)
    blocks[:, idx, idx] = block_diags
    return blocks


def dense_feature_gram_to_block_diag(
    dense_feature_gram: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Project a dense FEATURE_GRAM to padded block-diagonal storage.

    This drops cross-block feature correlations while preserving within-block
    correlations.
    """

    if dense_feature_gram.ndim != 2 or dense_feature_gram.shape[-1] != dense_feature_gram.shape[-2]:
        raise ValueError("dense_feature_gram must be square [p, p]")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    feature_dim = dense_feature_gram.shape[-1]
    padded_dim = ((feature_dim + block_size - 1) // block_size) * block_size
    if padded_dim != feature_dim:
        dense_feature_gram = torch.nn.functional.pad(
            dense_feature_gram, (0, padded_dim - feature_dim, 0, padded_dim - feature_dim)
        )
    num_blocks = padded_dim // block_size
    blocks = torch.empty(
        (num_blocks, block_size, block_size),
        device=dense_feature_gram.device,
        dtype=dense_feature_gram.dtype,
    )
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        blocks[block_idx].copy_(dense_feature_gram[start:end, start:end])
    return blocks


def block_diag_feature_gram_to_dense(
    block_feature_gram: torch.Tensor,
    *,
    feature_dim: int | None = None,
) -> torch.Tensor:
    """Materialize padded block-diagonal storage as a dense matrix.

    This is primarily for testing, diagnostics, and reference implementations.
    Production matrix rules should consume block storage directly.
    """

    if block_feature_gram.ndim != 3 or block_feature_gram.shape[-1] != block_feature_gram.shape[-2]:
        raise ValueError("block_feature_gram must have shape [num_blocks, b, b]")
    num_blocks, block_size, _ = block_feature_gram.shape
    padded_dim = num_blocks * block_size
    dense = torch.zeros(
        (padded_dim, padded_dim),
        device=block_feature_gram.device,
        dtype=block_feature_gram.dtype,
    )
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        dense[start:end, start:end] = block_feature_gram[block_idx]
    if feature_dim is not None:
        return dense[:feature_dim, :feature_dim]
    return dense


@dataclass(frozen=True)
class FeatureGramFactorization:
    """Reusable solve factor for ``G @ (C + ridge I)^-1``.

    The factorization is intentionally small and representation-aware:
    diagonal Grams cache the positive denominator, dense Grams cache either a
    Cholesky factor or the regularized dense Gram for fallback solves, and
    block-diagonal Grams cache batched block factors. Callers can cache this
    object across optimizer steps when FEATURE_GRAM refresh cadence is > 1.
    """

    kind: Literal[
        "diag",
        "dense_cholesky",
        "dense_fallback",
        "block_cholesky",
        "block_fallback",
    ]
    factor: torch.Tensor

    def right_solve(self, grad: torch.Tensor) -> torch.Tensor:
        """Solve ``out @ C_reg = grad`` using the cached factor."""

        compute_dtype = _matrix_solve_dtype(self.factor.dtype)
        grad = grad.to(compute_dtype)
        factor = self.factor.to(compute_dtype)
        if self.kind == "diag":
            return grad / factor
        if self.kind == "dense_cholesky":
            return torch.cholesky_solve(grad.mT, factor).mT
        if self.kind == "dense_fallback":
            return torch.linalg.solve(factor, grad.mT).mT
        if self.kind == "block_cholesky":
            return _right_solve_block_diag_from_factor(grad, factor, cholesky=True)
        if self.kind == "block_fallback":
            return _right_solve_block_diag_from_factor(grad, factor, cholesky=False)
        raise RuntimeError(f"Unsupported FeatureGramFactorization kind: {self.kind}")


def _pad_feature_axis(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    feature_dim = tensor.shape[-1]
    padded_dim = ((feature_dim + block_size - 1) // block_size) * block_size
    pad = padded_dim - feature_dim
    if pad == 0:
        return tensor, feature_dim
    return torch.nn.functional.pad(tensor, (0, pad)), feature_dim


def _right_solve_block_diag_from_factor(
    grad: torch.Tensor,
    block_factor: torch.Tensor,
    *,
    cholesky: bool,
) -> torch.Tensor:
    if block_factor.ndim != 3 or block_factor.shape[-1] != block_factor.shape[-2]:
        raise ValueError("block-diagonal solve factor must have shape [num_blocks, b, b]")
    block_size = block_factor.shape[-1]
    grad_padded, feature_dim = _pad_feature_axis(grad, block_size)
    num_blocks = block_factor.shape[0]
    expected_features = num_blocks * block_size
    if grad_padded.shape[-1] != expected_features:
        raise ValueError(
            f"block-diagonal feature_gram expects {expected_features} features, "
            f"got {grad.shape[-1]}."
        )

    q = grad_padded.shape[-2]
    grad_blocks = grad_padded.reshape(q, num_blocks, block_size).permute(1, 2, 0)
    if cholesky:
        solved = torch.cholesky_solve(grad_blocks, block_factor)
    else:
        solved = torch.linalg.solve(block_factor, grad_blocks)
    out = solved.permute(2, 0, 1).reshape(q, expected_features)
    return out[..., :feature_dim]


def _right_multiply_block_diag(grad: torch.Tensor, block_matrix: torch.Tensor) -> torch.Tensor:
    if block_matrix.ndim != 3 or block_matrix.shape[-1] != block_matrix.shape[-2]:
        raise ValueError("block_matrix must have shape [num_blocks, b, b]")
    block_size = block_matrix.shape[-1]
    grad_padded, feature_dim = _pad_feature_axis(grad, block_size)
    q = grad_padded.shape[-2]
    num_blocks = block_matrix.shape[0]
    expected_features = num_blocks * block_size
    if grad_padded.shape[-1] != expected_features:
        raise ValueError(
            f"block matrix expects {expected_features} features, got {grad.shape[-1]}."
        )
    grad_blocks = grad_padded.reshape(q, num_blocks, block_size).permute(1, 0, 2)
    out = torch.bmm(grad_blocks, block_matrix).permute(1, 0, 2).reshape(q, expected_features)
    return out[..., :feature_dim]


def _muon_scale_factor(size_out: int, size_in: int, mode: MuonScaleT) -> float:
    if mode == "none":
        return 1.0
    if mode == "shape_scaling":
        return max(1.0, size_out / size_in) ** 0.5
    if mode == "spectral":
        return max(size_out, size_in) ** 0.5
    if mode == "unit_rms_norm":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid Muon scale mode: {mode}")


def right_precondition_with_feature_gram(
    grad: torch.Tensor,
    feature_gram: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Return ``grad @ (feature_gram + ridge I)^-1``.

    A one-dimensional ``feature_gram`` is interpreted as a diagonal Gram. A
    three-dimensional ``feature_gram`` is interpreted as padded block-diagonal
    storage ``[num_blocks, block, block]``.
    """

    return factorize_feature_gram(feature_gram, ridge=ridge).right_solve(grad)


def factorize_feature_gram(
    feature_gram: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> FeatureGramFactorization:
    """Build a reusable right-solve factor for a supported FEATURE_GRAM."""

    c = _regularize_feature_gram(feature_gram, ridge)
    compute_dtype = _matrix_solve_dtype(c.dtype)
    c = c.to(compute_dtype)
    if c.ndim == 1:
        if torch.any(c <= 0):
            raise ValueError(
                "Diagonal feature_gram entries must be positive after ridge regularization."
            )
        return FeatureGramFactorization("diag", c)
    if c.ndim == 2:
        chol, info = torch.linalg.cholesky_ex(c)
        if torch.all(info == 0):
            return FeatureGramFactorization("dense_cholesky", chol)
        return FeatureGramFactorization("dense_fallback", c)
    if c.ndim == 3:
        chol, info = torch.linalg.cholesky_ex(c)
        if torch.all(info == 0):
            return FeatureGramFactorization("block_cholesky", chol)
        return FeatureGramFactorization("block_fallback", c)
    raise ValueError(
        "feature_gram must be diagonal [p], dense [p, p], or block-diagonal [num_blocks, b, b]"
    )


def right_precondition_with_factorized_feature_gram(
    grad: torch.Tensor,
    factorization: FeatureGramFactorization,
) -> torch.Tensor:
    """Return ``grad @ C_reg^-1`` using a cached FEATURE_GRAM factorization."""

    return factorization.right_solve(grad)


def locoprop_s_update(
    grad: torch.Tensor,
    feature_gram: torch.Tensor,
    *,
    gamma: float = 1.0,
    inner_lr: float | None = None,
    inner_steps: int | None = None,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Affine/quadratic LocoProp-S delta from ``(G, C)``.

    If ``inner_steps`` is ``None``, this returns the converged regularized
    update ``-gamma * G @ (C + ridge I)^-1``. Otherwise it returns the finite
    inner-loop delta
    ``-inner_lr * gamma * G @ sum_j (I - inner_lr C)^j``.
    """

    if inner_steps is None:
        return -gamma * right_precondition_with_feature_gram(grad, feature_gram, ridge=ridge)
    if inner_steps < 1:
        raise ValueError("inner_steps must be >= 1 when provided")
    if inner_lr is None:
        raise ValueError("inner_lr is required for finite-step LocoProp-S")

    c = _regularize_feature_gram(feature_gram, ridge)
    compute_dtype = _matrix_solve_dtype(c.dtype)
    c = c.to(compute_dtype)
    grad = grad.to(compute_dtype)

    if c.ndim == 1:
        powers = torch.ones_like(c)
        running = torch.ones_like(c)
        base = 1.0 - inner_lr * c
        for _ in range(1, inner_steps):
            running = running * base
            powers = powers + running
        return -inner_lr * gamma * grad * powers

    if c.ndim == 3:
        identity = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype).expand_as(c)
        base = identity - inner_lr * c
        powers = identity.clone()
        running = identity.clone()
        for _ in range(1, inner_steps):
            running = torch.bmm(running, base)
            powers = powers + running
        return -inner_lr * gamma * _right_multiply_block_diag(grad, powers)

    identity = torch.eye(c.shape[-1], device=c.device, dtype=c.dtype)
    base = identity - inner_lr * c
    powers = identity.clone()
    running = identity.clone()
    for _ in range(1, inner_steps):
        running = running.matmul(base)
        powers = powers + running
    return -inner_lr * gamma * grad.matmul(powers)


def newton_muon_update(
    grad: torch.Tensor,
    feature_gram: torch.Tensor,
    *,
    ridge: float = 0.0,
    num_ns_steps: int = 5,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    scale_mode: MuonScaleT = "spectral",
    extra_scale_factor: float = 1.0,
) -> torch.Tensor:
    """Newton-Muon-style local addable delta from ``(G, C)``.

    This is the backend-local mathematical rule. Distributed TP exactness is a
    caller responsibility: callers must only pass a matrix operand matching the
    declared TP apply mode.
    """

    preconditioned = right_precondition_with_feature_gram(grad, feature_gram, ridge=ridge)
    orthogonalized = newton_schulz_orthogonalize(
        preconditioned.to(torch.float32),
        steps=num_ns_steps,
        coefficient_type=coefficient_type,
        custom_coefficient_sets=custom_coefficient_sets,
    )
    scale = _muon_scale_factor(grad.size(-2), grad.size(-1), scale_mode)
    return -orthogonalized * scale * extra_scale_factor


def apply_diag_newton_muon_update_(
    param: torch.Tensor,
    grad: torch.Tensor,
    diag_feature_gram: torch.Tensor,
    *,
    lr: float,
    ridge: float = 0.0,
    num_ns_steps: int = 5,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    scale_mode: MuonScaleT = "spectral",
    extra_scale_factor: float = 1.0,
    weight_decay: float = 0.0,
    decoupled_weight_decay: bool = True,
    fp32_matmul_prec: str = "medium",
    use_syrk: bool = False,
) -> torch.Tensor:
    """Apply the diagonal Newton-Muon update in-place.

    This is the fast diagonal FEATURE_GRAM Newton-Muon path:
    diagonal right preconditioning feeds the shared Polar Express/NS Muon
    implementation, and the scaled parameter update is applied in-place. Dense
    and block-diagonal FEATURE_GRAM variants intentionally stay on the generic
    solve path.
    """

    if param.ndim != 2 or grad.ndim != 2:
        raise ValueError("param and grad must be two-dimensional")
    if param.shape != grad.shape:
        raise ValueError("param and grad must have matching shapes")
    if diag_feature_gram.ndim != 1:
        raise ValueError("diag_feature_gram must be one-dimensional")
    if diag_feature_gram.shape[0] != grad.shape[-1]:
        raise ValueError("diag_feature_gram length must match the parameter feature dimension")

    from emerging_optimizers.triton_kernels.diag_gram import (
        apply_matrix_update_kernel_,
        diag_right_precondition_matrix,
    )

    preconditioned = diag_right_precondition_matrix(
        grad,
        diag_feature_gram,
        param=param,
        ridge=ridge,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )

    with fp32_matmul_precision(fp32_matmul_prec):  # type: ignore[arg-type]
        orthogonalized = newton_schulz_orthogonalize(
            preconditioned.to(torch.float32),
            steps=num_ns_steps,
            coefficient_type=coefficient_type,
            custom_coefficient_sets=custom_coefficient_sets,
            use_syrk=use_syrk,
        )

    scale = _muon_scale_factor(grad.size(-2), grad.size(-1), scale_mode) * extra_scale_factor
    return apply_matrix_update_kernel_(
        param,
        orthogonalized,
        lr=lr,
        update_scale=scale,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


def apply_diag_right_preconditioned_update_(
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
    """Apply a diagonal right-preconditioned update in-place.

    This is the fused diagonal LocoProp-S/Newton-Muon preconditioning building
    block: optional weight decay, ``grad / (diag(C) + ridge)``, update scale,
    learning rate, and parameter add are performed without materializing a
    dense feature Gram. It is intentionally limited to diagonal factors.
    """

    if diag_feature_gram.ndim != 1:
        raise ValueError("diag_feature_gram must be one-dimensional")
    from emerging_optimizers.triton_kernels.diag_gram import (
        apply_diag_right_preconditioned_update_kernel_,
    )

    return apply_diag_right_preconditioned_update_kernel_(
        param,
        grad,
        diag_feature_gram,
        lr=lr,
        ridge=ridge,
        update_scale=update_scale,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


def apply_diag_left_preconditioned_update_(
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
    """Apply a diagonal left-preconditioned update in-place.

    This is the output-side analogue of diagonal FEATURE_GRAM right
    preconditioning: optional weight decay, ``diag(dY.T @ dY)`` row scaling,
    update scale, learning rate, and parameter add are performed without
    materializing a dense output Gram.
    """

    if diag_grad_gram.ndim != 1:
        raise ValueError("diag_grad_gram must be one-dimensional")
    from emerging_optimizers.triton_kernels.diag_gram import (
        apply_diag_left_preconditioned_update_kernel_,
    )

    return apply_diag_left_preconditioned_update_kernel_(
        param,
        grad,
        diag_grad_gram,
        lr=lr,
        ridge=ridge,
        update_scale=update_scale,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


def apply_diag_two_sided_preconditioned_update_(
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
    """Apply a diagonal left-and-right-preconditioned update in-place.

    This is the hot two-sided diagonal SGD path used when both output-side
    ``grad_gram`` and input-side ``feature_gram`` preconditioners are active.
    It avoids materializing the intermediate left-preconditioned direction and
    avoids a second parameter-update kernel launch on CUDA.
    """

    if diag_left.ndim != 1 or diag_right.ndim != 1:
        raise ValueError("diag_left and diag_right must be one-dimensional")
    from emerging_optimizers.triton_kernels.diag_gram import (
        apply_diag_two_sided_preconditioned_update_kernel_,
    )

    return apply_diag_two_sided_preconditioned_update_kernel_(
        param,
        grad,
        diag_left,
        diag_right,
        lr=lr,
        ridge_left=ridge_left,
        ridge_right=ridge_right,
        update_scale=update_scale,
        weight_decay=weight_decay,
        decoupled_weight_decay=decoupled_weight_decay,
    )


def newton_schulz_orthogonalize_grouped(
    matrices: Sequence[torch.Tensor],
    *,
    steps: int,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
) -> list[torch.Tensor]:
    """Run Newton-Schulz on same-shaped matrices as batched GEMMs.

    Matrices are grouped by ``(shape, dtype, device)`` and stacked, so same-shape
    transformer blocks use batched matmuls rather than one Python launch chain
    per parameter. This is a generic Torch implementation; CUDA-specific grouped
    GEMM kernels can replace this helper without changing callers.
    """

    groups: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[int, torch.Tensor]]] = {}
    for index, matrix in enumerate(matrices):
        groups.setdefault((tuple(matrix.shape), matrix.dtype, matrix.device), []).append((index, matrix))

    outputs: list[torch.Tensor | None] = [None] * len(matrices)
    for _, indexed in groups.items():
        batch = torch.stack([matrix for _, matrix in indexed], dim=0)
        batch_out = newton_schulz_orthogonalize(
            batch,
            steps=steps,
            coefficient_type=coefficient_type,
            custom_coefficient_sets=custom_coefficient_sets,
            eps=eps,
        )
        for batch_index, (original_index, _) in enumerate(indexed):
            outputs[original_index] = batch_out[batch_index]
    return [out for out in outputs if out is not None]
