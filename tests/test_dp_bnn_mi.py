"""
Unit tests for Mutual Information estimators:
DP-BNN, DP-MINE, MINE, NWJ, InfoNCE, and KSG.
"""

import unittest
import numpy as np
import torch

from src.envs.mi_benchmark_datasets import (
    gaussian_true_mi,
    generate_correlated_gaussian,
    generate_cubic,
    generate_sinusoid,
    generate_circular
)
from src.models.dp_bnn_mi import (
    KSG_Estimator,
    MINE_Estimator,
    NWJ_Estimator,
    InfoNCE_Estimator,
    DPMINE_Estimator,
    DPBNN_MIEstimator
)


class TestMIEstimators(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_gaussian_analytical_formula(self):
        # rho = 0 -> MI = 0
        self.assertAlmostEqual(gaussian_true_mi(dim=1, rho=0.0), 0.0)
        # rho = 0.6, dim = 1 -> -0.5 * ln(1 - 0.36) = -0.5 * ln(0.64) = 0.22314355
        expected = -0.5 * np.log(0.64)
        self.assertAlmostEqual(gaussian_true_mi(dim=1, rho=0.6), expected, places=5)
        # dim = 5 -> 5 * expected
        self.assertAlmostEqual(gaussian_true_mi(dim=5, rho=0.6), 5 * expected, places=5)

    def test_dataset_generation(self):
        X, Y, true_mi = generate_correlated_gaussian(n_samples=200, dim=2, rho=0.7, seed=42)
        self.assertEqual(X.shape, (200, 2))
        self.assertEqual(Y.shape, (200, 2))
        self.assertGreater(true_mi, 0.0)

        X_c, Y_c, mi_c = generate_cubic(n_samples=200, noise_std=0.3, seed=42)
        self.assertEqual(X_c.shape, (200, 1))
        self.assertGreater(mi_c, 0.0)

        X_s, Y_s, mi_s = generate_sinusoid(n_samples=200, noise_std=0.2, seed=42)
        self.assertEqual(X_s.shape, (200, 1))
        self.assertGreater(mi_s, 0.0)

        X_circ, Y_circ, mi_circ = generate_circular(n_samples=200, seed=42)
        self.assertEqual(X_circ.shape, (200, 1))
        self.assertGreater(mi_circ, 0.0)

    def test_ksg_estimator(self):
        # High correlation Gaussian: rho = 0.9, true MI = -0.5 * ln(1 - 0.81) = 0.83
        X, Y, true_mi = generate_correlated_gaussian(n_samples=1000, dim=1, rho=0.9, seed=42)
        ksg = KSG_Estimator(k=3)
        est = ksg.estimate(X, Y)
        # KSG should be close to true MI (within 0.2 nats)
        self.assertAlmostEqual(est, true_mi, delta=0.2)

    def test_neural_variational_estimators(self):
        dim_x = 2
        dim_y = 2
        x = torch.randn(64, dim_x)
        y = torch.randn(64, dim_y)

        # MINE
        mine = MINE_Estimator(dim_x, dim_y, hidden_dim=32)
        val_mine = mine.step(x, y)
        est_mine = mine.estimate(x, y)
        self.assertIsInstance(val_mine, float)
        self.assertIsInstance(est_mine, float)

        # NWJ
        nwj = NWJ_Estimator(dim_x, dim_y, hidden_dim=32)
        val_nwj = nwj.step(x, y)
        est_nwj = nwj.estimate(x, y)
        self.assertIsInstance(val_nwj, float)
        self.assertIsInstance(est_nwj, float)

        # InfoNCE
        nce = InfoNCE_Estimator(dim_x, dim_y, hidden_dim=32)
        val_nce = nce.step(x, y)
        est_nce = nce.estimate(x, y)
        self.assertIsInstance(val_nce, float)
        self.assertIsInstance(est_nce, float)

        # DP-MINE
        dpmine = DPMINE_Estimator(dim_x, dim_y, num_particles=3, hidden_dim=32)
        val_dpm = dpmine.step(x, y)
        est_dpm = dpmine.estimate(x, y)
        self.assertIsInstance(val_dpm, float)
        self.assertIsInstance(est_dpm, float)

    def test_dp_bnn_mi_estimator(self):
        dim_x = 2
        dim_y = 2
        x = torch.randn(64, dim_x)
        y = torch.randn(64, dim_y)

        dp_bnn = DPBNN_MIEstimator(dim_x, dim_y, num_particles=4, hidden_dim=32, symmetric=True)
        # Check train step runs and returns float loss
        loss = dp_bnn.train_step(x, y)
        self.assertIsInstance(loss, float)

        # Check estimate runs without error
        est = dp_bnn.estimate(x, y)
        self.assertIsInstance(est, float)
        self.assertGreaterEqual(est, 0.0)


if __name__ == "__main__":
    unittest.main()
