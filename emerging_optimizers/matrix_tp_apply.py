# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel apply helpers for matrix-function updates.

These helpers are intentionally explicit about exactness. Local block application
is exposed as an approximation; exact TP paths either all-gather the logical
matrix or use the supported small-Gram polar all-reduce sides.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from emerging_optimizers.matrix_update_rules import newton_schulz_orthogonalize

TPLayout = Literal["none", "duplicated", "column_parallel", "row_parallel"]
SmallGramSide = Literal["right", "left"]
SmallGramOrientation = SmallGramSide

__all__ = [
    "TPLayout",
    "SmallGramSide",
    "SmallGramOrientation",
    "allgather_logical_matrix",
    "shard_logical_matrix_like",
    "small_gram_newton_schulz_side",
    "small_gram_newton_schulz_orientation",
    "tp_small_gram_newton_schulz_allreduce",
    "supports_small_gram_polar_allreduce",
    "tp_allgather_logical_matrix_update",
    "tp_block_local_approx",
    "tp_small_gram_polar_allreduce",
]


def _dist_world_size(group: torch.distributed.ProcessGroup | None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def _dist_rank(group: torch.distributed.ProcessGroup | None) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(group=group)


def _shard_dim(tp_layout: TPLayout) -> int | None:
    if tp_layout == "column_parallel":
        return 0
    if tp_layout == "row_parallel":
        return 1
    if tp_layout in ("none", "duplicated"):
        return None
    raise ValueError(f"Unsupported TP layout: {tp_layout}")


def _require_2d_matrix(matrix: torch.Tensor, name: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {tuple(matrix.shape)}")


def allgather_logical_matrix(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-gather a TP-sharded matrix into its logical matrix.

    ``column_parallel`` is row-sharded and gathers along dim 0.
    ``row_parallel`` is column-sharded and gathers along dim 1.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    shard_dim = _shard_dim(tp_layout)
    world_size = _dist_world_size(group)
    if shard_dim is None or world_size == 1:
        return local_matrix

    gathered = [torch.empty_like(local_matrix) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local_matrix, group=group)
    return torch.cat(gathered, dim=shard_dim)


def shard_logical_matrix_like(
    logical_matrix: torch.Tensor,
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Return the shard of ``logical_matrix`` corresponding to ``local_matrix``."""

    _require_2d_matrix(logical_matrix, "logical_matrix")
    _require_2d_matrix(local_matrix, "local_matrix")
    shard_dim = _shard_dim(tp_layout)
    world_size = _dist_world_size(group)
    if shard_dim is None or world_size == 1:
        if tuple(logical_matrix.shape) != tuple(local_matrix.shape):
            raise ValueError(
                f"Logical matrix shape {tuple(logical_matrix.shape)} does not match local "
                f"matrix shape {tuple(local_matrix.shape)} for unsharded TP layout."
            )
        return logical_matrix

    rank = _dist_rank(group)
    expected = local_matrix.shape[shard_dim]
    expected_shape = list(local_matrix.shape)
    expected_shape[shard_dim] *= world_size
    if tuple(logical_matrix.shape) != tuple(expected_shape):
        raise ValueError(
            f"Logical matrix shape {tuple(logical_matrix.shape)} does not match expected "
            f"TP-gathered shape {tuple(expected_shape)}."
        )
    start = rank * expected
    end = start + expected
    index = [slice(None)] * logical_matrix.ndim
    index[shard_dim] = slice(start, end)
    return logical_matrix[tuple(index)].contiguous()


def tp_allgather_logical_matrix_update(
    local_matrix: torch.Tensor,
    update_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    """Exact reference path: all-gather, apply, then shard back."""

    _require_2d_matrix(local_matrix, "local_matrix")
    logical = allgather_logical_matrix(local_matrix, tp_layout=tp_layout, group=group)
    logical_update = update_fn(logical)
    return shard_logical_matrix_like(
        logical_update, local_matrix, tp_layout=tp_layout, group=group
    )


def _matrix_inverse_sqrt_psd(matrix: torch.Tensor, *, eps: float = 0.0) -> torch.Tensor:
    if eps < 0.0:
        raise ValueError("eps must be >= 0")
    matrix_fp32 = matrix.to(torch.float32)
    evals, evecs = torch.linalg.eigh(matrix_fp32)
    positive = evals > eps
    inv_sqrt = torch.where(positive, evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(), 0.0)
    return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT


def _matrix_inverse_sqrt_newton_schulz(
    matrix: torch.Tensor,
    *,
    steps: int,
    ridge: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Approximate ``matrix^-1/2`` using PyTorch/cuBLAS-backed Gram NS.

    This is a reference-performance backend, not a fused kernel. It keeps the
    iteration on the small symmetric Gram so row/column-sharded Muon can avoid
    all-gathering the full logical matrix during the optimizer step.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    _require_2d_matrix(matrix, "matrix")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix inverse square root requires a square matrix")

    gram = matrix.to(torch.float32)
    gram = 0.5 * (gram + gram.mT)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    if ridge != 0.0:
        gram = gram + ridge * eye

    scale = torch.linalg.matrix_norm(gram, ord="fro").clamp_min(eps)
    y = gram / scale
    z = eye
    for _ in range(steps):
        t = torch.addmm(3.0 * eye, z, y, beta=1.0, alpha=-1.0).mul_(0.5)
        y = y @ t
        z = t @ z
    return z / torch.sqrt(scale)


def supports_small_gram_polar_allreduce(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> bool:
    """Return whether the exact small-Gram polar all-reduce side is valid."""

    _require_2d_matrix(local_matrix, "local_matrix")
    world_size = _dist_world_size(group)
    rows, cols = local_matrix.shape[-2:]
    if tp_layout in ("none", "duplicated") or world_size == 1:
        return True
    if tp_layout == "column_parallel":
        logical_rows = rows * world_size
        return logical_rows >= cols
    if tp_layout == "row_parallel":
        logical_cols = cols * world_size
        return logical_cols >= rows
    return False


def small_gram_newton_schulz_side(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> SmallGramSide:
    """Return the exact small-Gram NS side for this local matrix.

    ``"right"`` means the exact Gram is ``M.T @ M`` and the update is
    ``M @ G^{-1/2}``; ``"left"`` means the exact Gram is ``M @ M.T`` and the
    update is ``G^{-1/2} @ M``. Distributed layouts must align with the matrix
    axis opposite the small Gram.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    world_size = _dist_world_size(group)
    rows, cols = local_matrix.shape[-2:]
    if world_size == 1 or tp_layout in ("none", "duplicated"):
        return "right" if rows >= cols else "left"
    if tp_layout == "column_parallel":
        logical_rows = rows * world_size
        if logical_rows < cols:
            raise ValueError(
                "small-Gram NS all-reduce for column_parallel requires a tall logical matrix."
            )
        return "right"
    if tp_layout == "row_parallel":
        logical_cols = cols * world_size
        if logical_cols < rows:
            raise ValueError(
                "small-Gram NS all-reduce for row_parallel requires a wide logical matrix."
            )
        return "left"
    raise ValueError(f"Unsupported TP layout: {tp_layout}")


def small_gram_newton_schulz_orientation(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> SmallGramSide:
    """Compatibility alias for ``small_gram_newton_schulz_side``."""

    return small_gram_newton_schulz_side(local_matrix, tp_layout=tp_layout, group=group)


def tp_small_gram_polar_allreduce(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """Polar update for supported TP shard sides.

    Supported exact cases:
    - ``column_parallel`` row shards where the logical matrix is tall, using
      all-reduced ``M.T @ M`` and local right multiplication.
    - ``row_parallel`` column shards where the logical matrix is wide, using
      all-reduced ``M @ M.T`` and local left multiplication.

    With the default ``eps=0``, this computes the small-Gram polar factor using
    a zero inverse on the Gram nullspace. Positive ``eps`` explicitly requests
    an epsilon-thresholded approximation.

    Unsupported sides must use the all-gather reference or an explicit
    block-local approximation.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    world_size = _dist_world_size(group)
    if world_size == 1 or tp_layout in ("none", "duplicated"):
        transpose = local_matrix.size(-2) > local_matrix.size(-1)
        matrix = local_matrix.to(torch.float32)
        if transpose:
            gram = matrix.mT @ matrix
            return (matrix @ _matrix_inverse_sqrt_psd(gram, eps=eps)).to(torch.float32)
        gram = matrix @ matrix.mT
        return (_matrix_inverse_sqrt_psd(gram, eps=eps) @ matrix).to(torch.float32)

    matrix = local_matrix.to(torch.float32)
    if tp_layout == "column_parallel":
        logical_rows = matrix.shape[-2] * world_size
        if logical_rows < matrix.shape[-1]:
            raise ValueError(
                "small_gram_polar all-reduce for column_parallel requires a tall logical matrix; "
                "use tp_allgather_logical_matrix or explicit block_local approximation."
            )
        gram = matrix.mT @ matrix
        torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
        return (matrix @ _matrix_inverse_sqrt_psd(gram, eps=eps)).to(torch.float32)

    if tp_layout == "row_parallel":
        logical_cols = matrix.shape[-1] * world_size
        if logical_cols < matrix.shape[-2]:
            raise ValueError(
                "small_gram_polar all-reduce for row_parallel requires a wide logical matrix; "
                "use tp_allgather_logical_matrix or explicit block_local approximation."
            )
        gram = matrix @ matrix.mT
        torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
        return (_matrix_inverse_sqrt_psd(gram, eps=eps) @ matrix).to(torch.float32)

    raise ValueError(f"Unsupported TP layout: {tp_layout}")


def tp_small_gram_newton_schulz_allreduce(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
    steps: int = 8,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """Reference row/column-sharded Muon update using all-reduced small Gram NS.

    ``column_parallel`` weights are row-sharded, so the exact small Gram is
    ``sum_i M_i.T @ M_i`` and each rank applies the right factor locally.
    ``row_parallel`` weights are column-sharded, so the exact small Gram is
    ``sum_i M_i @ M_i.T`` and each rank applies the left factor locally.

    This helper is intended as the semantic/reference path for matrix-aware
    FSDP and TP parity tests. It uses PyTorch matmul/addmm so CUDA execution is
    backed by cuBLAS/cuBLASLt, while avoiding full-matrix all-gather.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    world_size = _dist_world_size(group)
    matrix = local_matrix.to(torch.float32)

    side = small_gram_newton_schulz_side(
        local_matrix, tp_layout=tp_layout, group=group
    )

    if world_size == 1 or tp_layout in ("none", "duplicated"):
        if side == "right":
            gram = matrix.mT @ matrix
            factor = _matrix_inverse_sqrt_newton_schulz(gram, steps=steps, ridge=ridge)
            return (matrix @ factor).to(torch.float32)
        gram = matrix @ matrix.mT
        factor = _matrix_inverse_sqrt_newton_schulz(gram, steps=steps, ridge=ridge)
        return (factor @ matrix).to(torch.float32)

    if tp_layout == "column_parallel":
        gram = matrix.mT @ matrix
        torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
        factor = _matrix_inverse_sqrt_newton_schulz(gram, steps=steps, ridge=ridge)
        return (matrix @ factor).to(torch.float32)

    if tp_layout == "row_parallel":
        gram = matrix @ matrix.mT
        torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
        factor = _matrix_inverse_sqrt_newton_schulz(gram, steps=steps, ridge=ridge)
        return (factor @ matrix).to(torch.float32)

    raise ValueError(f"Unsupported TP layout: {tp_layout}")


def tp_block_local_approx(
    local_matrix: torch.Tensor,
    update_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    approximation_label: str | None = None,
) -> torch.Tensor:
    """Explicitly local TP approximation.

    Callers must carry/log ``approximation_label`` in their apply plan; this
    helper refuses unlabeled use so local NS/polar cannot masquerade as exact TP.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    if not approximation_label:
        raise ValueError("TP_BLOCK_LOCAL_APPROX requires a non-empty approximation_label")
    if update_fn is None:
        return newton_schulz_orthogonalize(local_matrix.to(torch.float32), steps=5)
    return update_fn(local_matrix)
