# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from absl.testing import absltest

from emerging_optimizers.triton_kernels.feature_gram import (
    apply_diag_left_preconditioned_update_kernel_,
    apply_diag_right_preconditioned_update_kernel_,
    apply_diag_two_sided_preconditioned_update_kernel_,
    diag_feature_gram_reduce,
    diag_grad_gram_reduce,
    diag_left_precondition_matrix,
)


class FeatureGramKernelFallbackTest(absltest.TestCase):
    def test_diag_feature_gram_reduce_fallback_matches_torch(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]])
        count = torch.zeros((), dtype=torch.float64)

        out = diag_feature_gram_reduce(
            x,
            count=count,
            mean=True,
            ridge=0.5,
            reciprocal=True,
        )

        expected = ((x * x).sum(dim=0) / x.shape[0] + 0.5).reciprocal()
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(count, torch.tensor(float(x.shape[0]), dtype=torch.float64))

    def test_diag_feature_gram_reduce_accumulates_into_existing_buffer(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        out = torch.ones(3)

        diag_feature_gram_reduce(x, out=out, accumulate=True)

        torch.testing.assert_close(out, torch.tensor([2.0, 5.0, 10.0]))

    def test_diag_grad_gram_reduce_fallback_matches_torch(self):
        dy = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
        count = torch.zeros((), dtype=torch.float64)

        out = diag_grad_gram_reduce(dy, count=count, mean=True, ridge=0.25)

        expected = (dy * dy).sum(dim=0) / dy.shape[0] + 0.25
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(count, torch.tensor(float(dy.shape[0]), dtype=torch.float64))

    def test_diag_left_precondition_matrix_fallback_matches_reference(self):
        param = torch.ones(2, 3)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
        diag = torch.tensor([1.0, 3.0])

        update = diag_left_precondition_matrix(
            grad,
            diag,
            param=param,
            ridge=1.0,
            weight_decay=0.2,
            decoupled_weight_decay=False,
        )

        expected = (grad + 0.2 * param) / (diag + 1.0)[:, None]
        torch.testing.assert_close(update, expected)

    def test_diag_right_update_kernel_fallback_matches_reference(self):
        param = torch.ones(2, 3)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
        diag = torch.tensor([1.0, 3.0, 5.0])

        apply_diag_right_preconditioned_update_kernel_(
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

    def test_diag_left_update_kernel_fallback_matches_reference(self):
        param = torch.ones(2, 3)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
        diag = torch.tensor([1.0, 3.0])

        apply_diag_left_preconditioned_update_kernel_(
            param,
            grad,
            diag,
            lr=0.1,
            ridge=1.0,
            update_scale=0.5,
            weight_decay=0.2,
            decoupled_weight_decay=True,
        )

        expected = torch.ones(2, 3) * 0.98 - 0.05 * grad / (diag + 1.0)[:, None]
        torch.testing.assert_close(param, expected)

    def test_diag_two_sided_update_kernel_fallback_matches_reference(self):
        param = torch.ones(2, 3)
        grad = torch.tensor([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
        diag_left = torch.tensor([1.0, 3.0])
        diag_right = torch.tensor([1.0, 3.0, 5.0])

        apply_diag_two_sided_preconditioned_update_kernel_(
            param,
            grad,
            diag_left,
            diag_right,
            lr=0.1,
            ridge_left=1.0,
            ridge_right=2.0,
            update_scale=0.5,
            weight_decay=0.2,
            decoupled_weight_decay=True,
        )

        denom = (diag_left + 1.0)[:, None] * (diag_right + 2.0)[None, :]
        expected = torch.ones(2, 3) * 0.98 - 0.05 * grad / denom
        torch.testing.assert_close(param, expected)


if __name__ == "__main__":
    absltest.main()
