# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from absl.testing import absltest

from emerging_optimizers.matrix_update_rules import (
    apply_diag_newton_muon_update_,
    apply_diag_right_preconditioned_update_,
    block_diag_feature_gram_to_dense,
    dense_feature_gram_to_block_diag,
    diag_feature_gram_to_block_diag,
    factorize_feature_gram,
    feature_gram_to_diag,
    locoprop_s_update,
    newton_muon_update,
    newton_schulz_orthogonalize_grouped,
    newton_schulz_orthogonalize,
    right_precondition_with_feature_gram,
    right_precondition_with_factorized_feature_gram,
)


class MatrixUpdateRulesTest(absltest.TestCase):
    def test_locoprop_s_finite_steps_matches_explicit_inner_loop_delta(self):
        x = torch.tensor([[1.0, 2.0], [3.0, -1.0], [2.0, 0.5]])
        delta = torch.tensor([[0.5, -0.25], [1.0, 0.75], [-0.5, 0.25]])
        w0 = torch.tensor([[0.2, -0.3], [0.4, 0.1]])
        gamma = 0.7
        inner_lr = 0.05
        inner_steps = 4

        grad = delta.mT.matmul(x)
        feature_gram = x.mT.matmul(x)
        target_cross = w0.matmul(feature_gram) - gamma * grad

        wk = w0.clone()
        for _ in range(inner_steps):
            wk = wk - inner_lr * (wk.matmul(feature_gram) - target_cross)
        explicit_delta = wk - w0

        gc_delta = locoprop_s_update(
            grad,
            feature_gram,
            gamma=gamma,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )

        torch.testing.assert_close(gc_delta, explicit_delta)

    def test_locoprop_s_delta_is_independent_of_base_weight(self):
        x = torch.tensor([[1.0, -0.5], [2.0, 0.25], [-1.5, 1.0], [0.5, 2.0]])
        delta = torch.tensor([[0.25, -0.75], [1.25, 0.5], [-0.5, 1.0], [0.75, -0.25]])
        base_weights = [
            torch.tensor([[0.2, -0.3], [0.4, 0.1]]),
            torch.tensor([[-2.0, 1.5], [0.25, 3.0]]),
        ]
        gamma = 0.4
        inner_lr = 0.03
        inner_steps = 5

        grad = delta.mT.matmul(x)
        feature_gram = x.mT.matmul(x)
        gc_delta = locoprop_s_update(
            grad,
            feature_gram,
            gamma=gamma,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
        )

        explicit_deltas = []
        for w0 in base_weights:
            target_cross = w0.matmul(feature_gram) - gamma * grad
            wk = w0.clone()
            for _ in range(inner_steps):
                wk = wk - inner_lr * (wk.matmul(feature_gram) - target_cross)
            explicit_deltas.append(wk - w0)

        torch.testing.assert_close(explicit_deltas[0], explicit_deltas[1])
        torch.testing.assert_close(gc_delta, explicit_deltas[0])

    def test_locoprop_s_converged_matches_right_solve(self):
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        feature_gram = torch.tensor([[3.0, 0.5], [0.5, 2.0]])
        gamma = 0.25

        update = locoprop_s_update(grad, feature_gram, gamma=gamma)
        ref = -gamma * torch.linalg.solve(feature_gram, grad.mT).mT

        torch.testing.assert_close(update, ref)

    def test_locoprop_s_converged_zeroes_regularized_local_residual(self):
        grad = torch.tensor([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]])
        feature_gram = torch.tensor(
            [[4.0, 0.5, -0.25], [0.5, 3.0, 0.75], [-0.25, 0.75, 2.0]]
        )
        gamma = 0.6
        ridge = 0.2

        update = locoprop_s_update(grad, feature_gram, gamma=gamma, ridge=ridge)
        c_reg = feature_gram + ridge * torch.eye(feature_gram.shape[-1])

        torch.testing.assert_close(
            update.matmul(c_reg) + gamma * grad,
            torch.zeros_like(grad),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_locoprop_s_finite_steps_uses_regularized_feature_gram(self):
        grad = torch.tensor([[1.0, 2.0]])
        feature_gram = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
        ridge = 0.5
        inner_lr = 0.1
        inner_steps = 3
        c = feature_gram + ridge * torch.eye(2)
        identity = torch.eye(2)
        powers = identity + (identity - inner_lr * c) + (identity - inner_lr * c).matmul(
            identity - inner_lr * c
        )

        update = locoprop_s_update(
            grad,
            feature_gram,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            ridge=ridge,
        )

        torch.testing.assert_close(update, -inner_lr * grad.matmul(powers))

    def test_locoprop_s_finite_steps_casts_grad_to_gram_dtype(self):
        grad = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        feature_gram = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float32)

        update = locoprop_s_update(
            grad,
            feature_gram,
            inner_lr=0.1,
            inner_steps=2,
            ridge=0.01,
        )

        self.assertEqual(update.dtype, torch.float32)
        self.assertTrue(torch.isfinite(update).all())

    def test_dense_right_precondition_solves_low_precision_gram_in_fp32(self):
        grad = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
        feature_gram = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.bfloat16)

        update = right_precondition_with_feature_gram(grad, feature_gram, ridge=0.01)

        self.assertEqual(update.dtype, torch.float32)
        self.assertTrue(torch.isfinite(update).all())

    def test_factorized_dense_feature_gram_matches_uncached_right_precondition(self):
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        feature_gram = torch.tensor([[3.0, 0.5], [0.5, 2.0]])

        factorization = factorize_feature_gram(feature_gram, ridge=0.1)

        self.assertEqual(factorization.kind, "dense_cholesky")
        torch.testing.assert_close(
            right_precondition_with_factorized_feature_gram(grad, factorization),
            right_precondition_with_feature_gram(grad, feature_gram, ridge=0.1),
        )

    def test_factorized_block_diag_feature_gram_matches_uncached_right_precondition(self):
        grad = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
        blocks = torch.tensor(
            [
                [[3.0, 0.5], [0.5, 2.0]],
                [[4.0, 0.0], [0.0, 1.0]],
            ]
        )

        factorization = factorize_feature_gram(blocks, ridge=0.1)

        self.assertEqual(factorization.kind, "block_cholesky")
        torch.testing.assert_close(
            right_precondition_with_factorized_feature_gram(grad, factorization),
            right_precondition_with_feature_gram(grad, blocks, ridge=0.1),
        )

    def test_factorized_diag_feature_gram_caches_regularized_denominator(self):
        grad = torch.tensor([[2.0, 8.0], [4.0, 16.0]])
        feature_gram = torch.tensor([2.0, 4.0])

        factorization = factorize_feature_gram(feature_gram, ridge=1.0)

        self.assertEqual(factorization.kind, "diag")
        torch.testing.assert_close(factorization.factor, torch.tensor([3.0, 5.0]))
        torch.testing.assert_close(
            right_precondition_with_factorized_feature_gram(grad, factorization),
            grad / torch.tensor([3.0, 5.0]),
        )

    def test_right_precondition_diag_feature_gram(self):
        grad = torch.tensor([[2.0, 8.0], [4.0, 16.0]])
        feature_gram = torch.tensor([2.0, 4.0])

        torch.testing.assert_close(
            right_precondition_with_feature_gram(grad, feature_gram),
            torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
        )

    def test_right_precondition_block_diag_matches_dense_block_matrix(self):
        grad = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
        blocks = torch.tensor(
            [
                [[3.0, 0.5], [0.5, 2.0]],
                [[4.0, 0.0], [0.0, 1.0]],
            ]
        )
        dense = torch.zeros(4, 4)
        dense[:2, :2] = blocks[0]
        dense[2:, 2:] = blocks[1]
        padded_grad = torch.nn.functional.pad(grad, (0, 1))

        actual = right_precondition_with_feature_gram(grad, blocks)
        expected = torch.linalg.solve(dense, padded_grad.mT).mT[:, :3]

        torch.testing.assert_close(actual, expected)

    def test_diag_feature_gram_to_block_diag_is_lossless_storage_embedding(self):
        diag = torch.tensor([1.0, 2.0, 3.0])

        blocks = diag_feature_gram_to_block_diag(diag, block_size=2)

        self.assertEqual(tuple(blocks.shape), (2, 2, 2))
        torch.testing.assert_close(feature_gram_to_diag(blocks, feature_dim=3), diag)
        torch.testing.assert_close(
            block_diag_feature_gram_to_dense(blocks, feature_dim=3),
            torch.diag(diag),
        )

    def test_block_diag_to_diag_drops_only_off_diagonal_block_correlations(self):
        blocks = torch.tensor(
            [
                [[2.0, 0.25], [0.25, 3.0]],
                [[4.0, -0.5], [-0.5, 5.0]],
            ]
        )

        diag = feature_gram_to_diag(blocks, feature_dim=3)

        torch.testing.assert_close(diag, torch.tensor([2.0, 3.0, 4.0]))

    def test_dense_feature_gram_to_block_diag_preserves_within_block_entries(self):
        dense = torch.tensor(
            [
                [1.0, 0.1, 0.2],
                [0.1, 2.0, 0.3],
                [0.2, 0.3, 3.0],
            ]
        )

        blocks = dense_feature_gram_to_block_diag(dense, block_size=2)
        projected = block_diag_feature_gram_to_dense(blocks, feature_dim=3)

        expected = torch.tensor(
            [
                [1.0, 0.1, 0.0],
                [0.1, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
        torch.testing.assert_close(projected, expected)

    def test_locoprop_s_block_diag_matches_dense_converged_update(self):
        grad = torch.tensor([[1.0, 2.0, 3.0]])
        blocks = torch.tensor(
            [
                [[2.0, 0.25], [0.25, 1.5]],
                [[3.0, 0.0], [0.0, 1.0]],
            ]
        )
        dense = torch.zeros(4, 4)
        dense[:2, :2] = blocks[0]
        dense[2:, 2:] = blocks[1]
        padded_grad = torch.nn.functional.pad(grad, (0, 1))
        gamma = 0.7
        ridge = 0.1

        actual = locoprop_s_update(grad, blocks, gamma=gamma, ridge=ridge)
        expected = -gamma * torch.linalg.solve(
            dense + ridge * torch.eye(4), padded_grad.mT
        ).mT[:, :3]

        torch.testing.assert_close(actual, expected)

    def test_locoprop_s_block_diag_finite_steps_matches_dense_blocks(self):
        grad = torch.tensor([[1.0, -0.5, 2.0]])
        blocks = torch.tensor(
            [
                [[1.5, 0.25], [0.25, 1.25]],
                [[2.0, 0.0], [0.0, 1.0]],
            ]
        )
        dense = torch.zeros(4, 4)
        dense[:2, :2] = blocks[0]
        dense[2:, 2:] = blocks[1]
        padded_grad = torch.nn.functional.pad(grad, (0, 1))
        inner_lr = 0.05
        inner_steps = 4
        identity = torch.eye(4)
        base = identity - inner_lr * dense
        powers = identity.clone()
        running = identity.clone()
        for _ in range(1, inner_steps):
            running = running @ base
            powers = powers + running

        actual = locoprop_s_update(
            grad, blocks, inner_lr=inner_lr, inner_steps=inner_steps
        )
        expected = -inner_lr * padded_grad.matmul(powers)[:, :3]

        torch.testing.assert_close(actual, expected)

    def test_apply_diag_right_preconditioned_update_inplace(self):
        param = torch.ones(2, 3)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
        diag = torch.tensor([1.0, 3.0, 5.0])

        apply_diag_right_preconditioned_update_(
            param,
            grad,
            diag,
            lr=0.1,
            ridge=1.0,
            update_scale=0.5,
            weight_decay=0.2,
            decoupled_weight_decay=True,
        )

        expected = torch.ones(2, 3) * 0.98 - 0.05 * grad / (diag + 1.0)
        torch.testing.assert_close(param, expected)

    def test_right_precondition_diag_feature_gram_rejects_singular_without_ridge(self):
        grad = torch.tensor([[2.0, 8.0]])
        feature_gram = torch.tensor([2.0, 0.0])

        with self.assertRaisesRegex(ValueError, "positive"):
            right_precondition_with_feature_gram(grad, feature_gram)

        torch.testing.assert_close(
            right_precondition_with_feature_gram(grad, feature_gram, ridge=1.0),
            torch.tensor([[2.0 / 3.0, 8.0]]),
        )

    def test_newton_muon_dense_feature_gram_preconditions_before_orthogonalization(self):
        grad = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [3.0, -1.0]], dtype=torch.float32)
        feature_gram = torch.tensor([[3.0, 0.4], [0.4, 1.5]], dtype=torch.float32)
        ridge = 0.1

        update = newton_muon_update(
            grad,
            feature_gram,
            ridge=ridge,
            num_ns_steps=3,
            coefficient_type="simple",
            scale_mode="none",
        )
        preconditioned = torch.linalg.solve(
            feature_gram + ridge * torch.eye(2), grad.mT
        ).mT
        ref = -newton_schulz_orthogonalize(
            preconditioned, steps=3, coefficient_type="simple"
        )

        torch.testing.assert_close(update, ref)

    def test_newton_muon_identity_feature_gram_matches_scaled_muon(self):
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        feature_gram = torch.eye(2)

        update = newton_muon_update(
            grad,
            feature_gram,
            num_ns_steps=1,
            coefficient_type="simple",
        )
        ref = newton_schulz_orthogonalize(grad, steps=1, coefficient_type="simple")

        torch.testing.assert_close(update, -ref * (max(grad.shape) ** 0.5))

    def test_apply_diag_newton_muon_update_matches_out_of_place_rule(self):
        param = torch.ones(2, 3, dtype=torch.float32)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
        diag = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
        lr = 0.1
        weight_decay = 0.2

        expected = param.clone()
        expected.mul_(1.0 - lr * weight_decay)
        expected.add_(
            newton_muon_update(
                grad,
                diag,
                ridge=1.0,
                num_ns_steps=2,
                coefficient_type="simple",
                scale_mode="none",
            ),
            alpha=lr,
        )

        apply_diag_newton_muon_update_(
            param,
            grad,
            diag,
            lr=lr,
            ridge=1.0,
            num_ns_steps=2,
            coefficient_type="simple",
            scale_mode="none",
            weight_decay=weight_decay,
            decoupled_weight_decay=True,
            fp32_matmul_prec="highest",
            use_syrk=False,
        )

        torch.testing.assert_close(param, expected)

    def test_newton_schulz_grouped_matches_per_matrix_results(self):
        matrices = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[0.5, -1.0], [2.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        ]

        grouped = newton_schulz_orthogonalize_grouped(
            matrices, steps=2, coefficient_type="simple"
        )
        refs = [
            newton_schulz_orthogonalize(m, steps=2, coefficient_type="simple")
            for m in matrices
        ]

        for actual, expected in zip(grouped, refs):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    absltest.main()
