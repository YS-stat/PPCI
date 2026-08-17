from __future__ import annotations

import unittest
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ppci_condmean.data import generate_simulation_labeled, nw_oracle_mean, prepare_blogfeedback_ppci, simulation_predictor
from ppci_condmean.estimator import (
    fit_ppci_mean,
    lo_mean_from_weights,
    ppci_mean_from_weight_values,
    ppci_plus_mean_from_weight_values,
    ppci_plus_mean_given_omegas,
)
from ppci_condmean.diagnostics import nw_closeness_diagnostics
from ppci_condmean.joint_tuning import (
    JointTuningConfig,
    collect_joint_candidate_cache,
    lambda_grid_joint,
    select_joint_from_cache,
    tune_joint_from_covariates,
)
from ppci_condmean.utils import source_sha256
from ppci_condmean.weights import RKHSLocalizationWeight


class CoreSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)
        self.X = self.rng.uniform(size=(80, 3))
        self.x0 = np.array([0.75, 0.75, 0.75])

    def test_source_fingerprint_is_deterministic(self) -> None:
        first = source_sha256()
        second = source_sha256()
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 64)
        self.assertGreater(first[1], 0)

    def test_rkhs_localization_weight_is_finite(self) -> None:
        weight = RKHSLocalizationWeight(self.X, self.x0, h=0.8, lam=0.01, kernel="matern52")
        values = weight(self.X[:12])
        self.assertEqual(values.shape, (12,))
        self.assertTrue(np.isfinite(values).all())

    def test_rkhs_localization_training_identity(self) -> None:
        weight = RKHSLocalizationWeight(self.X, self.x0, h=0.8, lam=0.01, kernel="matern52")
        np.testing.assert_allclose(weight(self.X), weight.training_values(), rtol=2e-10, atol=2e-10)

    def test_nw_closeness_diagnostics_are_finite(self) -> None:
        weight = RKHSLocalizationWeight(self.X, self.x0, h=0.8, lam=10.0, kernel="matern52")
        diagnostic = nw_closeness_diagnostics(weight)
        self.assertTrue(np.isfinite(list(diagnostic.values())).all())
        self.assertGreater(diagnostic["nw_corr"], 0.99)
        self.assertGreaterEqual(diagnostic["negative_weight_fraction"], 0.0)
        self.assertLessEqual(diagnostic["negative_weight_fraction"], 1.0)

    def test_p1_tuning_returns_grid_candidate(self) -> None:
        config = JointTuningConfig(
            h_factors=(0.8, 1.0),
            lambda_factor_min=0.1,
            lambda_factor_max=2.0,
            lambda_grid_size=5,
            bias_screen="p1_label",
            c_bias=0.5,
            constraint_fallback="least_violation",
            backend="cpu",
        )
        result = tune_joint_from_covariates(self.X, self.x0, n=60, method="GH", cfg=config)
        self.assertIn(result.h_factor_vs_median, config.h_factors)
        self.assertGreater(result.lam, 0.0)
        self.assertTrue(np.isfinite(result.sw_proxy))

    def test_shrinking_lambda_grid_matches_paper(self) -> None:
        n = 200
        config = JointTuningConfig(
            lambda_factor_min=0.2,
            lambda_factor_max=2.0,
            lambda_grid_size=3,
            lambda_grid_mode="shrinking",
        )
        expected = np.logspace(np.log10(0.2), np.log10(2.0), 3)
        expected /= n * np.log(np.log(n + np.e**np.e))
        np.testing.assert_allclose(lambda_grid_joint(n, config), expected)

    def test_candidate_J_Q_and_bias_score_recompute_from_weights(self) -> None:
        n = 60
        config = JointTuningConfig(
            h_factors=(0.8,), lambda_factor_min=0.1, lambda_factor_max=0.2,
            lambda_grid_size=2, bias_screen="p1_label", backend="cpu",
        )
        cache = collect_joint_candidate_cache(self.X, self.x0, n=n, cfg=config)
        row = cache["candidate_pool"][0][0]
        weights = np.asarray(row["w_train"], dtype=float)
        q_hat = float(np.mean(weights**2))
        expected_score = np.sqrt(
            n * row["lambda"] * max(row["D_h_point"] - q_hat, 0.0) / max(q_hat, 1e-12)
        )
        self.assertAlmostEqual(row["J_w"], float(np.mean(weights)))
        self.assertAlmostEqual(row["Q_h"], q_hat)
        self.assertAlmostEqual(row["V_w"], q_hat)
        self.assertAlmostEqual(row["bias_score_label"], expected_score)

    def test_empty_bias_feasible_set_uses_least_violation(self) -> None:
        config = JointTuningConfig(
            h_factors=(0.8,), lambda_factor_min=0.1, lambda_factor_max=0.2,
            lambda_grid_size=2, tau_op=100.0, tau_loc=100.0,
            bias_screen="p1_label", c_bias=0.0,
            constraint_fallback="least_violation", backend="cpu",
        )
        cache = collect_joint_candidate_cache(self.X, self.x0, n=60, cfg=config)
        result = select_joint_from_cache(cache, "GH", cfg=config)
        self.assertEqual(result.status, "fallback_least_normalized_violation")
        self.assertEqual(result.n_feasible_total, 0)

    def test_cpu_gpu_tuning_parity_when_cuda_is_available(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        common = dict(
            h_factors=(0.8, 1.0), lambda_factor_min=0.1,
            lambda_factor_max=1.0, lambda_grid_size=4,
            bias_screen="p1_label", c_bias=0.5,
        )
        cpu = tune_joint_from_covariates(
            self.X, self.x0, n=60, method="GH", cfg=JointTuningConfig(**common, backend="cpu")
        )
        gpu = tune_joint_from_covariates(
            self.X, self.x0, n=60, method="GH", cfg=JointTuningConfig(**common, backend="torch")
        )
        np.testing.assert_allclose(
            [cpu.h, cpu.lam, cpu.J_w, cpu.Q_h, cpu.bias_score],
            [gpu.h, gpu.lam, gpu.J_w, gpu.Q_h, gpu.bias_score],
            rtol=1e-8, atol=1e-10,
        )

    def test_mean_estimator_formula(self) -> None:
        Y_l = np.array([1.0, 3.0, 5.0])
        f_l = np.array([0.5, 2.0, 4.5])
        f_u = np.array([2.0, 4.0, 6.0, 8.0])
        w_l = np.array([1.0, 2.0, 1.0])
        w_u = np.array([1.0, 1.5, 0.5, 1.0])
        result = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u)
        expected = (np.mean(w_l * (Y_l - f_l)) + np.mean(w_u * f_u)) / np.mean(w_u)
        self.assertAlmostEqual(result.theta_hat, expected)
        self.assertTrue(np.isfinite(result.se))

    def test_simulation_predictor_is_fixed_function_of_x(self) -> None:
        X = self.rng.uniform(size=(40, 3))
        f = simulation_predictor(X, quality=0.75)
        for seed, sigma in [(11, 1.0), (23, 2.0)]:
            rng = np.random.default_rng(seed)
            X_draw, _, _, f_draw = generate_simulation_labeled(rng, 40, sigma_eps=sigma, predictor_quality=0.75)
            np.testing.assert_allclose(f_draw, simulation_predictor(X_draw, quality=0.75))
        np.testing.assert_allclose(f, simulation_predictor(X, quality=0.75))

    def test_ppci_plus_endpoints_match_lo_and_ppci(self) -> None:
        n, N = 20, 50
        Y_l = self.rng.normal(size=n)
        f_l = self.rng.normal(size=n)
        f_u = self.rng.normal(size=N)
        w_l = self.rng.uniform(0.5, 1.5, size=n)
        w_u = self.rng.uniform(0.5, 1.5, size=N)
        folds = np.arange(n) % 2
        lo = lo_mean_from_weights(np.empty((n, 0)), Y_l, w_l)
        ppci = ppci_mean_from_weight_values(Y_l, f_l, f_u, w_l, w_u)
        plus0 = ppci_plus_mean_given_omegas(Y_l, f_l, f_u, w_l, w_u, folds, np.zeros(2))
        plus1 = ppci_plus_mean_given_omegas(Y_l, f_l, f_u, w_l, w_u, folds, np.ones(2))
        np.testing.assert_allclose([plus0.theta_hat, plus0.se], [lo.theta_hat, lo.se])
        np.testing.assert_allclose([plus1.theta_hat, plus1.se], [ppci.theta_hat, ppci.se])

    def test_data_driven_ppci_plus_is_finite_and_clipped(self) -> None:
        n, N = 40, 90
        X_l = self.rng.uniform(size=(n, 3))
        X_u = self.rng.uniform(size=(N, 3))
        f_l = X_l.sum(axis=1)
        f_u = X_u.sum(axis=1)
        Y_l = f_l + self.rng.normal(scale=0.3, size=n)
        result = ppci_plus_mean_from_weight_values(
            Y_l, f_l, f_u, np.ones(n), np.ones(N), rng=np.random.default_rng(19), omega_folds=5
        )
        self.assertTrue(np.isfinite([result.theta_hat, result.se, result.J_hat, result.V_hat]).all())
        self.assertTrue(0.0 <= result.omega_1 <= 1.0)
        self.assertTrue(0.0 <= result.omega_2 <= 1.0)
        self.assertEqual(result.omega_folds, 5)
        self.assertTrue(result.omega_min <= result.omega <= result.omega_max)

    def test_blogfeedback_predictor_excludes_targets_by_default(self) -> None:
        defaults = prepare_blogfeedback_ppci.__defaults__
        self.assertIsNotNone(defaults)
        self.assertFalse(defaults[-2])

    def test_blogfeedback_reference_uses_heldout_inference_population(self) -> None:
        raw = self.rng.normal(size=(120, 5))
        raw[:, -1] = np.exp(0.2 * raw[:, 0] - 0.1 * raw[:, 1])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "blogData_train.csv"
            zip_path = Path(tmp) / "blogfeedback.zip"
            np.savetxt(csv_path, raw, delimiter=",")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.write(csv_path, arcname=csv_path.name)
            data = prepare_blogfeedback_ppci(
                zip_path,
                seed=31,
                n_x0=3,
                ppci_fraction=0.3,
                model="ridge",
            )
        expected = np.array([
            nw_oracle_mean(data["X_ppci"], data["Y_ppci"], x0, kernel="matern52")
            for x0 in data["x0"]
        ])
        np.testing.assert_allclose(data["theta0"], expected, rtol=0.0, atol=1e-12)
        self.assertEqual(data["reference_population"], "heldout_ppci_pool")

    def test_public_api_uses_joint_p1_tuning(self) -> None:
        X_l = self.rng.uniform(size=(24, 3))
        X_u = self.rng.uniform(size=(60, 3))
        f_l = X_l.sum(axis=1)
        f_u = X_u.sum(axis=1)
        Y_l = f_l + self.rng.normal(scale=0.5, size=len(f_l))
        config = JointTuningConfig(
            h_factors=(0.8, 1.0),
            lambda_factor_min=0.1,
            lambda_factor_max=1.0,
            lambda_grid_size=4,
            bias_screen="p1_label",
            c_bias=0.5,
            backend="cpu",
        )
        ppci, lo, ppi, info = fit_ppci_mean(
            X_l, Y_l, f_l, X_u, f_u, self.x0, seed=11, tuning_cfg=config
        )
        self.assertEqual(ppci.lambda_selection, "joint_p1")
        self.assertEqual(lo.lambda_selection, "joint_p1")
        self.assertEqual(info["tuning_1"]["bias_screen"], "p1_label")
        self.assertEqual(info["tuning_1"]["M"], len(X_u) // 2)
        self.assertEqual(info["tuning_2"]["M"], len(X_u) - len(X_u) // 2)
        self.assertTrue(np.isfinite([ppci.theta_hat, ppci.se, lo.theta_hat, ppi.theta_hat]).all())

    def test_nosplit_uses_all_observed_covariates_for_operator(self) -> None:
        X_l = self.rng.uniform(size=(24, 3))
        X_u = self.rng.uniform(size=(60, 3))
        f_l = X_l.sum(axis=1)
        f_u = X_u.sum(axis=1)
        Y_l = f_l + self.rng.normal(scale=0.5, size=len(f_l))
        config = JointTuningConfig(
            h_factors=(0.8, 1.0),
            lambda_factor_min=0.1,
            lambda_factor_max=1.0,
            lambda_grid_size=4,
            bias_screen="p1_label",
            c_bias=0.5,
            backend="cpu",
        )
        ppci, lo, ppi, info = fit_ppci_mean(
            X_l, Y_l, f_l, X_u, f_u, self.x0, split="nosplit", seed=11, tuning_cfg=config
        )
        self.assertEqual(ppci.lambda_selection, "joint_p1_nosplit")
        self.assertEqual(lo.lambda_selection, "joint_p1_nosplit")
        self.assertEqual(info["tuning"]["M"], len(X_l) + len(X_u))
        self.assertTrue(np.isfinite([ppci.theta_hat, ppci.se, lo.theta_hat, ppi.theta_hat]).all())


if __name__ == "__main__":
    unittest.main()
