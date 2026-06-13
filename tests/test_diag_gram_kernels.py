# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from absl.testing import absltest

from emerging_optimizers.triton_kernels.diag_gram import (
    HAS_TRITON_DIAG_GRAM,
    apply_diag_left_preconditioned_update_kernel_,
    apply_diag_right_preconditioned_update_kernel_,
    apply_diag_two_sided_preconditioned_update_kernel_,
    diag_feature_gram_reduce,
    diag_grad_gram_reduce,
    diag_left_precondition_matrix,
)


class FeatureGramKernelFallbackTest(absltest.TestCase):
    def _cuda_or_skip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if not HAS_TRITON_DIAG_GRAM:
            self.skipTest("Triton not available")
        return torch.device("cuda")

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

    def test_diag_update_kernels_cuda_match_reference(self):
        device = self._cuda_or_skip()
        grad = torch.arange(1, 1 + 17 * 65, device=device, dtype=torch.float32).reshape(17, 65)
        diag_left = torch.linspace(1.0, 3.0, 17, device=device)
        diag_right = torch.linspace(1.0, 5.0, 65, device=device)

        right_param = torch.ones_like(grad)
        apply_diag_right_preconditioned_update_kernel_(
            right_param,
            grad,
            diag_right,
            lr=0.01,
            ridge=0.5,
            update_scale=0.25,
            weight_decay=0.1,
            decoupled_weight_decay=True,
        )
        right_expected = torch.ones_like(grad) * 0.999 - 0.0025 * grad / (diag_right + 0.5)
        torch.testing.assert_close(right_param, right_expected)

        left_param = torch.ones_like(grad)
        apply_diag_left_preconditioned_update_kernel_(
            left_param,
            grad,
            diag_left,
            lr=0.01,
            ridge=0.25,
            update_scale=0.5,
            weight_decay=0.1,
            decoupled_weight_decay=True,
        )
        left_expected = torch.ones_like(grad) * 0.999 - 0.005 * grad / (diag_left + 0.25)[:, None]
        torch.testing.assert_close(left_param, left_expected)

        two_sided_param = torch.ones_like(grad)
        apply_diag_two_sided_preconditioned_update_kernel_(
            two_sided_param,
            grad,
            diag_left,
            diag_right,
            lr=0.01,
            ridge_left=0.25,
            ridge_right=0.5,
            update_scale=0.75,
            weight_decay=0.1,
            decoupled_weight_decay=True,
        )
        two_sided_expected = torch.ones_like(grad) * 0.999 - 0.0075 * grad / (
            (diag_left + 0.25)[:, None] * (diag_right + 0.5)[None, :]
        )
        torch.testing.assert_close(two_sided_param, two_sided_expected)


if __name__ == "__main__":
    absltest.main()
