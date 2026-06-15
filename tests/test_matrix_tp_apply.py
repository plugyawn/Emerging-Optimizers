# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import torch
from absl.testing import absltest

from emerging_optimizers.matrix_tp_apply import (
    allgather_logical_matrix,
    shard_logical_matrix_like,
    supports_small_gram_polar_allreduce,
    tp_allgather_logical_matrix_update,
    tp_block_local_approx,
    tp_small_gram_newton_schulz_allreduce,
    tp_small_gram_polar_allreduce,
)


@contextmanager
def fake_distributed(
    *,
    world_size=2,
    rank=0,
    all_gather_values=None,
    all_reduce_scale=None,
    all_reduce_value=None,
):
    old_is_available = torch.distributed.is_available
    old_is_initialized = torch.distributed.is_initialized
    old_get_world_size = torch.distributed.get_world_size
    old_get_rank = torch.distributed.get_rank
    old_all_gather = torch.distributed.all_gather
    old_all_reduce = torch.distributed.all_reduce

    def fake_all_gather(gathered, local, group=None):
        values = all_gather_values
        if values is None:
            values = [local + float(idx) for idx in range(world_size)]
        for dst, value in zip(gathered, values):
            dst.copy_(value)

    def fake_all_reduce(tensor, op=None, group=None):
        if all_reduce_value is not None:
            tensor.copy_(all_reduce_value)
            return
        if all_reduce_scale is not None:
            tensor.mul_(all_reduce_scale)

    torch.distributed.is_available = lambda: True
    torch.distributed.is_initialized = lambda: True
    torch.distributed.get_world_size = lambda group=None: world_size
    torch.distributed.get_rank = lambda group=None: rank
    torch.distributed.all_gather = fake_all_gather
    torch.distributed.all_reduce = fake_all_reduce
    try:
        yield
    finally:
        torch.distributed.is_available = old_is_available
        torch.distributed.is_initialized = old_is_initialized
        torch.distributed.get_world_size = old_get_world_size
        torch.distributed.get_rank = old_get_rank
        torch.distributed.all_gather = old_all_gather
        torch.distributed.all_reduce = old_all_reduce


class MatrixTPApplyTest(absltest.TestCase):
    def test_allgather_reference_degenerates_to_local_without_tp(self):
        local = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        update = tp_allgather_logical_matrix_update(
            local,
            lambda matrix: matrix + 1.0,
            tp_layout="none",
        )

        torch.testing.assert_close(update, local + 1.0)

    def test_allgather_reference_applies_update_to_full_column_parallel_matrix(self):
        local = torch.tensor([[1.0, 2.0]])
        values = [local, local + 10.0]

        with fake_distributed(world_size=2, rank=1, all_gather_values=values):
            update = tp_allgather_logical_matrix_update(
                local,
                lambda matrix: matrix + torch.tensor([[0.0, 0.0], [100.0, 200.0]]),
                tp_layout="column_parallel",
            )

        torch.testing.assert_close(update, values[1] + torch.tensor([[100.0, 200.0]]))

    def test_allgather_column_parallel_concatenates_rows_and_shards_back(self):
        local = torch.tensor([[1.0, 2.0]])
        values = [local, local + 10.0]
        with fake_distributed(world_size=2, rank=1, all_gather_values=values):
            logical = allgather_logical_matrix(local, tp_layout="column_parallel")
            shard = shard_logical_matrix_like(logical, local, tp_layout="column_parallel")

        torch.testing.assert_close(logical, torch.cat(values, dim=0))
        torch.testing.assert_close(shard, values[1])

    def test_allgather_row_parallel_concatenates_columns_and_shards_back(self):
        local = torch.tensor([[1.0], [2.0]])
        values = [local, local + 10.0]
        with fake_distributed(world_size=2, rank=1, all_gather_values=values):
            logical = allgather_logical_matrix(local, tp_layout="row_parallel")
            shard = shard_logical_matrix_like(logical, local, tp_layout="row_parallel")

        torch.testing.assert_close(logical, torch.cat(values, dim=1))
        torch.testing.assert_close(shard, values[1])

    def test_tp_helpers_reject_non_2d_matrices(self):
        local = torch.empty(2, 3, 4)

        with self.assertRaisesRegex(ValueError, "2D matrix"):
            allgather_logical_matrix(local, tp_layout="column_parallel")
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            shard_logical_matrix_like(local, local, tp_layout="column_parallel")
        with self.assertRaisesRegex(ValueError, "2D matrix"):
            tp_small_gram_polar_allreduce(local, tp_layout="none")

    def test_shard_logical_matrix_rejects_oversized_logical_matrix(self):
        local = torch.tensor([[1.0, 2.0]])
        oversized = torch.empty(3, 2)

        with fake_distributed(world_size=2, rank=0):
            with self.assertRaisesRegex(ValueError, "expected TP-gathered shape"):
                shard_logical_matrix_like(oversized, local, tp_layout="column_parallel")

    def test_small_gram_polar_matches_direct_polar_without_tp(self):
        matrix = torch.tensor([[1.0, 0.5], [0.25, 2.0], [1.5, -0.5]])
        gram = matrix.mT @ matrix
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = torch.where(
            evals > 0.0,
            evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(),
            0.0,
        )
        ref = matrix @ ((evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT)

        update = tp_small_gram_polar_allreduce(matrix, tp_layout="none")

        torch.testing.assert_close(update, ref)

    def test_small_gram_polar_column_parallel_matches_stacked_reference(self):
        local = torch.tensor([[1.0, 0.25], [0.5, 2.0]])
        full = torch.cat([local, local], dim=0)
        gram = full.mT @ full
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = torch.where(
            evals > 0.0,
            evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(),
            0.0,
        )
        ref = local @ ((evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT)

        with fake_distributed(world_size=2, rank=0, all_reduce_scale=2.0):
            update = tp_small_gram_polar_allreduce(local, tp_layout="column_parallel")

        torch.testing.assert_close(update, ref)

    def test_small_gram_polar_column_parallel_matches_nonidentical_shards(self):
        shards = [
            torch.tensor([[1.0, 0.25], [0.5, 2.0]]),
            torch.tensor([[-0.75, 1.25], [2.0, -1.0]]),
        ]
        full = torch.cat(shards, dim=0)
        gram = full.mT @ full
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = torch.where(
            evals > 0.0,
            evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(),
            0.0,
        )
        right_factor = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT

        for rank, local in enumerate(shards):
            with fake_distributed(world_size=2, rank=rank, all_reduce_value=gram):
                update = tp_small_gram_polar_allreduce(local, tp_layout="column_parallel")
            torch.testing.assert_close(update, local @ right_factor)

    def test_small_gram_polar_row_parallel_matches_stacked_reference(self):
        local = torch.tensor([[1.0, 0.25], [0.5, 2.0]])
        full = torch.cat([local, local], dim=1)
        gram = full @ full.mT
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = torch.where(
            evals > 0.0,
            evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(),
            0.0,
        )
        ref = ((evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT) @ local

        with fake_distributed(world_size=2, rank=0, all_reduce_scale=2.0):
            update = tp_small_gram_polar_allreduce(local, tp_layout="row_parallel")

        torch.testing.assert_close(update, ref)

    def test_small_gram_polar_row_parallel_matches_nonidentical_shards(self):
        shards = [
            torch.tensor([[1.0, 0.25], [0.5, 2.0]]),
            torch.tensor([[-0.75, 1.25], [2.0, -1.0]]),
        ]
        full = torch.cat(shards, dim=1)
        gram = full @ full.mT
        evals, evecs = torch.linalg.eigh(gram)
        inv_sqrt = torch.where(
            evals > 0.0,
            evals.clamp_min(torch.finfo(evals.dtype).tiny).rsqrt(),
            0.0,
        )
        left_factor = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.mT

        for rank, local in enumerate(shards):
            with fake_distributed(world_size=2, rank=rank, all_reduce_value=gram):
                update = tp_small_gram_polar_allreduce(local, tp_layout="row_parallel")
            torch.testing.assert_close(update, left_factor @ local)

    def test_small_gram_ns_column_parallel_matches_full_reference(self):
        local = torch.tensor([[1.0, 0.25], [0.5, 2.0], [1.25, -0.5]])
        shards = [local, local]
        full = torch.cat(shards, dim=0)
        full_update = tp_small_gram_newton_schulz_allreduce(
            full,
            tp_layout="none",
            steps=12,
            ridge=1e-5,
        )
        for rank, local in enumerate(shards):
            with fake_distributed(world_size=2, rank=rank, all_reduce_scale=2.0):
                update = tp_small_gram_newton_schulz_allreduce(
                    local,
                    tp_layout="column_parallel",
                    steps=12,
                    ridge=1e-5,
                )
            torch.testing.assert_close(update, full_update.chunk(2, dim=0)[rank])

    def test_small_gram_ns_row_parallel_matches_full_reference(self):
        local = torch.tensor([[1.0, 0.25, -0.5], [0.5, 2.0, 1.25]])
        shards = [local, local]
        full = torch.cat(shards, dim=1)
        full_update = tp_small_gram_newton_schulz_allreduce(
            full,
            tp_layout="none",
            steps=12,
            ridge=1e-5,
        )
        for rank, local in enumerate(shards):
            with fake_distributed(world_size=2, rank=rank, all_reduce_scale=2.0):
                update = tp_small_gram_newton_schulz_allreduce(
                    local,
                    tp_layout="row_parallel",
                    steps=12,
                    ridge=1e-5,
                )
            torch.testing.assert_close(update, full_update.chunk(2, dim=1)[rank])

    def test_small_gram_ns_accepts_explicit_logical_shape(self):
        local = torch.tensor([[1.0, 0.25, -0.5]])

        with fake_distributed(world_size=2, rank=0, all_reduce_scale=2.0):
            update = tp_small_gram_newton_schulz_allreduce(
                local,
                tp_layout="column_parallel",
                logical_shape=(4, 3),
                steps=2,
            )

        self.assertEqual(update.shape, local.shape)

    def test_small_gram_ns_rejects_incompatible_logical_shape(self):
        local = torch.tensor([[1.0, 0.25, -0.5]])

        with fake_distributed(world_size=2, rank=0):
            with self.assertRaisesRegex(ValueError, "preserve the local feature dimension"):
                tp_small_gram_newton_schulz_allreduce(
                    local,
                    tp_layout="column_parallel",
                    logical_shape=(4, 4),
                    steps=2,
                )

    def test_small_gram_ns_rejects_unsharded_logical_shape_mismatch(self):
        local = torch.tensor([[1.0, 0.25, -0.5]])

        with self.assertRaisesRegex(ValueError, "must match local_matrix shape"):
            tp_small_gram_newton_schulz_allreduce(
                local,
                tp_layout="none",
                logical_shape=(2, 3),
                steps=2,
            )

    def test_small_gram_rejects_unsupported_column_orientation(self):
        local = torch.empty(1, 4)
        with fake_distributed(world_size=2, rank=0):
            self.assertFalse(
                supports_small_gram_polar_allreduce(local, tp_layout="column_parallel")
            )
            with self.assertRaisesRegex(ValueError, "requires a tall logical matrix"):
                tp_small_gram_polar_allreduce(local, tp_layout="column_parallel")

    def test_small_gram_rejects_unsupported_row_orientation(self):
        local = torch.empty(4, 1)
        with fake_distributed(world_size=2, rank=0):
            self.assertFalse(supports_small_gram_polar_allreduce(local, tp_layout="row_parallel"))
            with self.assertRaisesRegex(ValueError, "requires a wide logical matrix"):
                tp_small_gram_polar_allreduce(local, tp_layout="row_parallel")

    def test_small_gram_support_without_tp(self):
        matrix = torch.empty(2, 3)

        self.assertTrue(supports_small_gram_polar_allreduce(matrix, tp_layout="none"))

    def test_block_local_requires_approximation_label(self):
        matrix = torch.empty(2, 2)

        with self.assertRaisesRegex(ValueError, "approximation_label"):
            tp_block_local_approx(matrix)

    def test_block_local_uses_explicit_update_fn_when_labeled(self):
        matrix = torch.tensor([[1.0, 2.0]])

        update = tp_block_local_approx(
            matrix,
            lambda local: local * 2.0,
            approximation_label="block_local_test",
        )

        torch.testing.assert_close(update, matrix * 2.0)


if __name__ == "__main__":
    absltest.main()
