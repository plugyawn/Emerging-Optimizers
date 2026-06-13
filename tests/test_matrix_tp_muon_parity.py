# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.multiprocessing as mp

from emerging_optimizers.matrix_tp_apply import (
    small_gram_newton_schulz_side,
    tp_small_gram_newton_schulz_allreduce,
)


class _FakeGroup:
    pass


def _patch_all_reduce_to_global_gram(monkeypatch, global_gram: torch.Tensor) -> _FakeGroup:
    group = _FakeGroup()

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        del op, async_op
        assert group is not None
        tensor.copy_(global_gram)
        return None

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    return group


def test_small_gram_muon_row_shards_match_full_local_update(monkeypatch):
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    reference = tp_small_gram_newton_schulz_allreduce(
        matrix,
        tp_layout="none",
        steps=12,
        ridge=1e-5,
    )
    group = _patch_all_reduce_to_global_gram(monkeypatch, matrix.t().matmul(matrix))

    sharded = torch.cat(
        [
            tp_small_gram_newton_schulz_allreduce(
                local_matrix,
                tp_layout="column_parallel",
                group=group,
                steps=12,
                ridge=1e-5,
            )
            for local_matrix in matrix.chunk(2, dim=0)
        ],
        dim=0,
    )

    torch.testing.assert_close(sharded, reference, rtol=1e-5, atol=1e-5)


def test_small_gram_muon_column_shards_match_full_local_update(monkeypatch):
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    reference = tp_small_gram_newton_schulz_allreduce(
        matrix,
        tp_layout="none",
        steps=12,
        ridge=1e-5,
    )
    group = _patch_all_reduce_to_global_gram(monkeypatch, matrix.matmul(matrix.t()))

    sharded = torch.cat(
        [
            tp_small_gram_newton_schulz_allreduce(
                local_matrix,
                tp_layout="row_parallel",
                group=group,
                steps=12,
                ridge=1e-5,
            )
            for local_matrix in matrix.chunk(2, dim=1)
        ],
        dim=1,
    )

    torch.testing.assert_close(sharded, reference, rtol=1e-5, atol=1e-5)


def test_small_gram_side_contract(monkeypatch):
    group = _patch_all_reduce_to_global_gram(monkeypatch, torch.eye(3))

    assert (
        small_gram_newton_schulz_side(
            torch.zeros(4, 3),
            tp_layout="column_parallel",
            group=group,
        )
        == "right"
    )
    assert (
        small_gram_newton_schulz_side(
            torch.zeros(3, 4),
            tp_layout="row_parallel",
            group=group,
        )
        == "left"
    )
    assert (
        small_gram_newton_schulz_side(torch.zeros(5, 2), tp_layout="none")
        == "right"
    )
    assert (
        small_gram_newton_schulz_side(torch.zeros(2, 5), tp_layout="none")
        == "left"
    )


def test_small_gram_side_rejects_wrong_aspect_ratio(monkeypatch):
    group = _patch_all_reduce_to_global_gram(monkeypatch, torch.eye(3))

    try:
        small_gram_newton_schulz_side(
            torch.zeros(1, 4),
            tp_layout="column_parallel",
            group=group,
        )
    except ValueError as exc:
        assert "tall logical matrix" in str(exc)
    else:
        raise AssertionError("column_parallel wide logical matrix should be rejected")

    try:
        small_gram_newton_schulz_side(
            torch.zeros(4, 1),
            tp_layout="row_parallel",
            group=group,
        )
    except ValueError as exc:
        assert "wide logical matrix" in str(exc)
    else:
        raise AssertionError("row_parallel tall logical matrix should be rejected")


def _distributed_small_gram_muon_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        row_sharded_tall = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        local_rows = row_sharded_tall.chunk(world_size, dim=0)[rank].contiguous()
        local_row_update = tp_small_gram_newton_schulz_allreduce(
            local_rows,
            tp_layout="column_parallel",
            steps=12,
            ridge=1e-5,
        )
        gathered_rows = [torch.empty_like(local_row_update) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_rows, local_row_update)
        row_update = torch.cat(gathered_rows, dim=0)
        row_reference = tp_small_gram_newton_schulz_allreduce(
            row_sharded_tall,
            tp_layout="none",
            steps=12,
            ridge=1e-5,
        )
        torch.testing.assert_close(row_update, row_reference, rtol=1e-5, atol=1e-5)

        column_sharded_wide = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        local_cols = column_sharded_wide.chunk(world_size, dim=1)[rank].contiguous()
        local_col_update = tp_small_gram_newton_schulz_allreduce(
            local_cols,
            tp_layout="row_parallel",
            steps=12,
            ridge=1e-5,
        )
        gathered_cols = [torch.empty_like(local_col_update) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_cols, local_col_update)
        col_update = torch.cat(gathered_cols, dim=1)
        col_reference = tp_small_gram_newton_schulz_allreduce(
            column_sharded_wide,
            tp_layout="none",
            steps=12,
            ridge=1e-5,
        )
        torch.testing.assert_close(col_update, col_reference, rtol=1e-5, atol=1e-5)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires torch.distributed with gloo",
)
def test_small_gram_muon_distributed_gloo_parity(tmp_path):
    init_file = os.fspath(tmp_path / "small_gram_muon_init")
    mp.start_processes(
        _distributed_small_gram_muon_worker,
        args=(2, init_file),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
