# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel apply helpers for matrix-function updates.

These helpers are intentionally explicit about exactness. Local block application
is exposed as an approximation; exact TP paths either all-gather the logical
matrix or use the supported small-Gram polar all-reduce sides.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import torch

from emerging_optimizers.matrix_update_rules import newton_schulz_orthogonalize
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
    _COEFFICIENT_SETS,
    get_coefficient_iterator,
)

TPLayout = Literal["none", "duplicated", "column_parallel", "row_parallel"]
SmallGramSide = Literal["right", "left"]
NSCoeffT = Literal["simple", "quintic", "polar_express", "cans", "aol", "custom"]

__all__ = [
    "TPLayout",
    "SmallGramSide",
    "allgather_logical_matrix",
    "shard_logical_matrix_like",
    "small_gram_newton_schulz_side",
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
    logical_shape: tuple[int, int] | None = None,
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
    if logical_shape is not None:
        if len(logical_shape) != 2:
            raise ValueError(f"logical_shape must be 2D, got {logical_shape}")
        logical_rows, logical_cols = logical_shape
        if tp_layout in ("none", "duplicated") and (logical_rows, logical_cols) != (rows, cols):
            raise ValueError(
                "unsharded small-Gram NS logical_shape must match local_matrix shape: "
                f"logical_shape={logical_shape}, local_shape={tuple(local_matrix.shape)}."
            )
        if tp_layout == "column_parallel" and logical_cols != cols:
            raise ValueError(
                "column_parallel small-Gram NS logical_shape must preserve the local "
                f"feature dimension: logical_shape={logical_shape}, local_shape={tuple(local_matrix.shape)}."
            )
        if tp_layout == "column_parallel" and logical_rows < rows:
            raise ValueError(
                "column_parallel small-Gram NS logical_shape cannot be smaller than the local "
                f"row shard: logical_shape={logical_shape}, local_shape={tuple(local_matrix.shape)}."
            )
        if tp_layout == "row_parallel" and logical_rows != rows:
            raise ValueError(
                "row_parallel small-Gram NS logical_shape must preserve the local "
                f"output dimension: logical_shape={logical_shape}, local_shape={tuple(local_matrix.shape)}."
            )
        if tp_layout == "row_parallel" and logical_cols < cols:
            raise ValueError(
                "row_parallel small-Gram NS logical_shape cannot be smaller than the local "
                f"column shard: logical_shape={logical_shape}, local_shape={tuple(local_matrix.shape)}."
            )
    else:
        logical_rows, logical_cols = rows, cols
        if world_size > 1 and tp_layout == "column_parallel":
            logical_rows = rows * world_size
        elif world_size > 1 and tp_layout == "row_parallel":
            logical_cols = cols * world_size
    if world_size == 1 or tp_layout in ("none", "duplicated"):
        return "right" if logical_rows >= logical_cols else "left"
    if tp_layout == "column_parallel":
        if logical_rows < logical_cols:
            raise ValueError(
                "small-Gram NS all-reduce for column_parallel requires a tall logical matrix."
            )
        return "right"
    if tp_layout == "row_parallel":
        if logical_cols < logical_rows:
            raise ValueError(
                "small-Gram NS all-reduce for row_parallel requires a wide logical matrix."
            )
        return "left"
    raise ValueError(f"Unsupported TP layout: {tp_layout}")


def _newton_schulz_coefficients(
    *,
    steps: int,
    coefficient_type: NSCoeffT,
    custom_coefficient_sets: Sequence[tuple[float, float, float]] | None,
):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError("custom_coefficient_sets must be provided when coefficient_type is 'custom'.")
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")
    iter_mode = "repeat_last" if coefficient_type in ("polar_express", "cans") else "cycle"
    return get_coefficient_iterator(steps, coefficient_sets, mode=iter_mode)


def _normalize_small_gram_matrix(
    matrix: torch.Tensor,
    *,
    distributed: bool,
    group: torch.distributed.ProcessGroup | None,
    eps: float,
) -> torch.Tensor:
    norm_sq = torch.sum(matrix * matrix)
    if distributed:
        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.SUM, group=group)
    return matrix / torch.sqrt(norm_sq).clamp_min(eps)


def _small_gram_newton_schulz_step(
    matrix: torch.Tensor,
    *,
    side: SmallGramSide,
    distributed: bool,
    group: torch.distributed.ProcessGroup | None,
    a: float,
    b: float,
    c: float,
    ridge: float,
) -> torch.Tensor:
    if side == "right":
        gram = matrix.mT @ matrix
        if distributed:
            torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
        if ridge != 0.0:
            gram = gram + ridge * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        gram_sq = gram @ gram
        return a * matrix + b * (matrix @ gram) + c * (matrix @ gram_sq)

    gram = matrix @ matrix.mT
    if distributed:
        torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM, group=group)
    if ridge != 0.0:
        gram = gram + ridge * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    gram_sq = gram @ gram
    return a * matrix + b * (gram @ matrix) + c * (gram_sq @ matrix)


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
    logical_shape: tuple[int, int] | None = None,
    steps: int = 8,
    coefficient_type: NSCoeffT = "quintic",
    custom_coefficient_sets: Sequence[tuple[float, float, float]] | None = None,
    ridge: float = 0.0,
    eps: float = 1e-7,
    use_syrk: bool = False,
) -> torch.Tensor:
    """Reference row/column-sharded Muon update using all-reduced small Gram NS.

    ``column_parallel`` weights are row-sharded, so the exact small Gram is
    ``sum_i M_i.T @ M_i`` and each rank applies the right factor locally.
    ``row_parallel`` weights are column-sharded, so the exact small Gram is
    ``sum_i M_i @ M_i.T`` and each rank applies the left factor locally.

    This helper is the semantic/reference path for TP parity tests and
    matrix-axis-aware DP/FSDP experiments. It applies the same Muon
    Newton-Schulz coefficient schedule as the full-matrix helper, but rewrites
    each polynomial step through the small Gram so CUDA execution is backed by
    PyTorch matmul/cuBLAS while avoiding full-matrix all-gather.
    """

    _require_2d_matrix(local_matrix, "local_matrix")
    if ridge < 0.0:
        raise ValueError("ridge must be >= 0")
    if eps <= 0.0:
        raise ValueError("eps must be > 0")
    if use_syrk:
        raise NotImplementedError(
            "tp_small_gram_newton_schulz_allreduce does not implement TSYRK-backed "
            "small-Gram substeps yet; pass use_syrk=False."
        )
    world_size = _dist_world_size(group)
    matrix = local_matrix.to(torch.float32)

    side = small_gram_newton_schulz_side(
        local_matrix,
        tp_layout=tp_layout,
        group=group,
        logical_shape=logical_shape,
    )
    distributed = world_size > 1 and tp_layout not in ("none", "duplicated")
    matrix = _normalize_small_gram_matrix(
        matrix,
        distributed=distributed,
        group=group,
        eps=eps,
    )
    coeff_iter = _newton_schulz_coefficients(
        steps=steps,
        coefficient_type=coefficient_type,
        custom_coefficient_sets=custom_coefficient_sets,
    )

    for a, b, c in coeff_iter:
        matrix = _small_gram_newton_schulz_step(
            matrix,
            side=side,
            distributed=distributed,
            group=group,
            a=a,
            b=b,
            c=c,
            ridge=ridge,
        )

    return matrix.to(torch.float32)


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
