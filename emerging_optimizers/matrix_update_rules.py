# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix update rules that consume affine wgrad factors.

The public functions in this module consume only the ordinary affine weight
gradient ``G = dY.T @ X`` and optional feature Gram ``C = X.T @ X``. They do
not require raw activations, LocoProp targets, output-feature crosses, or the
current logical weight as an operand for the update direction.
"""

from __future__ import annotations

from itertools import chain, cycle, islice, repeat
from typing import Iterator, Literal, Sequence

import torch

CoeffIterMode = Literal["cycle", "repeat_last"]
NSCoeffT = Literal["simple", "quintic", "polar_express", "cans", "aol", "custom"]
MuonScaleT = Literal["none", "shape_scaling", "spectral", "unit_rms_norm"]

__all__ = [
    "CoeffIterMode",
    "NSCoeffT",
    "MuonScaleT",
    "apply_diag_right_preconditioned_update_",
    "locoprop_s_update",
    "newton_muon_update",
    "newton_schulz_orthogonalize_grouped",
    "newton_schulz_orthogonalize",
    "right_precondition_with_feature_gram",
]

# Kept local to avoid importing the orthogonalized optimizer package on CPU-only
# test paths; that package imports Triton kernels at module import time.
_COEFFICIENT_SETS: dict[str, list[tuple[float, float, float]]] = {
    "simple": [(3.4445, -4.7750, 2.0315)],
    "quintic": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "cans": [
        (8.4703, -25.1081, 18.6293),
        (4.1828, -3.1087, 0.5806),
        (3.9619, -2.9541, 0.5630),
        (3.2866, -2.4647, 0.5074),
        (2.2737, -1.6447, 0.4162),
    ],
    "aol": [
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def _get_coefficient_iterator(
    steps: int,
    coefficient_sets: Sequence[tuple[float, float, float]],
    mode: CoeffIterMode = "cycle",
) -> Iterator[tuple[float, float, float]]:
    if not coefficient_sets:
        raise ValueError("coefficient_sets must be non-empty")
    if mode == "cycle":
        base: Iterator[tuple[float, float, float]] = cycle(coefficient_sets)
    elif mode == "repeat_last":
        base = chain(coefficient_sets, repeat(coefficient_sets[-1]))
    else:
        raise ValueError(f"Invalid coefficient iterator mode: {mode}")
    return islice(base, steps)


def newton_schulz_orthogonalize(
    x: torch.Tensor,
    *,
    steps: int,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
    transpose: bool | None = None,
) -> torch.Tensor:
    """Torch-only Newton-Schulz orthogonalization used by matrix rules."""

    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")
    if steps < 0:
        raise ValueError("steps must be >= 0")

    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    out = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)  # type: ignore[arg-type]

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError(
                "custom_coefficient_sets must be provided when coefficient_type is 'custom'"
            )
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    iter_mode: CoeffIterMode = (
        "repeat_last" if coefficient_type in ("polar_express", "cans") else "cycle"
    )
    for a, b, c in _get_coefficient_iterator(steps, coefficient_sets, mode=iter_mode):
        gram = out @ out.mT
        out = a * out + (b * gram + c * gram @ gram) @ out

    if transpose:
        out = out.mT
    return out.to(torch.float32)


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


def _right_solve_dense_spd(grad: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """Solve ``out @ gram = grad`` with a Cholesky fast path."""

    chol, info = torch.linalg.cholesky_ex(gram)
    if torch.all(info == 0):
        return torch.cholesky_solve(grad.mT, chol).mT
    return torch.linalg.solve(gram, grad.mT).mT


def _pad_feature_axis(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    feature_dim = tensor.shape[-1]
    padded_dim = ((feature_dim + block_size - 1) // block_size) * block_size
    pad = padded_dim - feature_dim
    if pad == 0:
        return tensor, feature_dim
    return torch.nn.functional.pad(tensor, (0, pad)), feature_dim


def _right_solve_block_diag_spd(grad: torch.Tensor, block_gram: torch.Tensor) -> torch.Tensor:
    """Solve ``out @ block_diag(block_gram) = grad``.

    ``block_gram`` uses padded fixed-size storage ``[num_blocks, block, block]``.
    Any padded feature columns in the last block are returned and then cropped
    back to ``grad.shape[-1]``.
    """

    if block_gram.ndim != 3 or block_gram.shape[-1] != block_gram.shape[-2]:
        raise ValueError("block-diagonal feature_gram must have shape [num_blocks, b, b]")
    block_size = block_gram.shape[-1]
    grad_padded, feature_dim = _pad_feature_axis(grad, block_size)
    num_blocks = block_gram.shape[0]
    expected_features = num_blocks * block_size
    if grad_padded.shape[-1] != expected_features:
        raise ValueError(
            f"block-diagonal feature_gram expects {expected_features} features, "
            f"got {grad.shape[-1]}."
        )

    q = grad_padded.shape[-2]
    grad_blocks = grad_padded.reshape(q, num_blocks, block_size).permute(1, 2, 0)
    chol, info = torch.linalg.cholesky_ex(block_gram)
    if torch.all(info == 0):
        solved = torch.cholesky_solve(grad_blocks, chol)
    else:
        solved = torch.linalg.solve(block_gram, grad_blocks)
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

    c = _regularize_feature_gram(feature_gram, ridge)
    compute_dtype = _matrix_solve_dtype(c.dtype)
    c = c.to(compute_dtype)
    grad = grad.to(compute_dtype)
    if c.ndim == 1:
        if torch.any(c <= 0):
            raise ValueError(
                "Diagonal feature_gram entries must be positive after ridge regularization."
            )
        return grad / c
    if c.ndim == 2:
        return _right_solve_dense_spd(grad, c)
    if c.ndim == 3:
        return _right_solve_block_diag_spd(grad, c)
    raise ValueError(
        "feature_gram must be diagonal [p], dense [p, p], or block-diagonal [num_blocks, b, b]"
    )


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
    if weight_decay != 0.0 and decoupled_weight_decay:
        param.mul_(1.0 - lr * weight_decay)
    update_grad = grad
    if weight_decay != 0.0 and not decoupled_weight_decay:
        update_grad = grad.add(param, alpha=weight_decay)
    denom = diag_feature_gram.to(update_grad.dtype) + ridge
    if torch.any(denom <= 0):
        raise ValueError("Diagonal feature_gram entries must be positive after ridge regularization.")
    param.add_(update_grad / denom, alpha=-lr * update_scale)
    return param


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
