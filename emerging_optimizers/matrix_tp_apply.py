# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel apply helpers for matrix-function updates.

These helpers are intentionally explicit about exactness. Local block application
is exposed as an approximation; exact TP paths either all-gather the logical
matrix or use the supported small-Gram polar all-reduce orientations.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from emerging_optimizers.matrix_update_rules import newton_schulz_orthogonalize

TPLayout = Literal["none", "duplicated", "column_parallel", "row_parallel"]

__all__ = [
    "TPLayout",
    "allgather_logical_matrix",
    "shard_logical_matrix_like",
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


def supports_small_gram_polar_allreduce(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
) -> bool:
    """Return whether the exact small-Gram polar all-reduce orientation is valid."""

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


def tp_small_gram_polar_allreduce(
    local_matrix: torch.Tensor,
    *,
    tp_layout: TPLayout,
    group: torch.distributed.ProcessGroup | None = None,
    eps: float = 0.0,
) -> torch.Tensor:
    """Polar update for supported TP shard orientations.

    Supported exact cases:
    - ``column_parallel`` row shards where the logical matrix is tall, using
      all-reduced ``M.T @ M`` and local right multiplication.
    - ``row_parallel`` column shards where the logical matrix is wide, using
      all-reduced ``M @ M.T`` and local left multiplication.

    With the default ``eps=0``, this computes the small-Gram polar factor using
    a zero inverse on the Gram nullspace. Positive ``eps`` explicitly requests
    an epsilon-thresholded approximation.

    Unsupported orientations must use the all-gather reference or an explicit
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
