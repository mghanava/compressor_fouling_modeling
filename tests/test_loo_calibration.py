"""
Unit tests for plot_loo_calibration_curve_with_reference and compute_loo_pit_model_agnostic.

Test strategy
-------------
Three tiers of tests:

1. **Analytical / unit tests** - feed known inputs (pre-computed PIT values,
   exact uniform samples) directly into the helpers and assert outputs are
   correct without touching PyMC.

2. **Oracle (self-consistent) PyMC model** - generate data *from* a model,
   then fit *that exact model* on it.  Because the model is correctly specified,
   LOO-PIT must be approximately Uniform(0,1) and the calibration curve must
   lie on the diagonal.  This is the gold-standard positive test.

3. **Deliberately miscalibrated PyMC models** - fit a model that is *wrong*
   (e.g. variance too small → over-confident → LOO-PIT will be S-shaped).
   These are negative tests: the function *must* detect miscalibration.
"""

from __future__ import annotations

import arviz_stats as azs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest
from scipy import stats
from scipy.stats import binom
from xarray import DataTree

from compressor_fouling_modeling.utility import (
    CalibrationStats,
    compute_loo_pit_model_agnostic,
    plot_loo_calibration_curve_with_reference,
)

# ============================================================================
# Helpers shared by multiple tests
# ============================================================================

RNG_SEED = 42


def _uniform_weights(n_obs: int, n_samples: int) -> np.ndarray:
    """All importance weights equal → plain empirical CDF."""
    return np.full((n_obs, n_samples), 1.0 / n_samples)


def _fit_oracle_normal(
    n: int = 200,
    true_mu: float = 0.0,
    true_sigma: float = 1.0,
    draws: int = 1000,
    chains: int = 2,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, DataTree]:
    """
    Oracle normal model: data ~ N(mu, sigma).

    We generate data from the exact likelihood and fit the same model, so the
    posterior predictive *must* be calibrated.

    Returns
    -------
    y_obs : (n,) observed data
    idata : ArviZ InferenceData with posterior_predictive and log_likelihood
    """
    rng = np.random.default_rng(seed)
    y_obs = rng.normal(true_mu, true_sigma, size=n)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

        idata = pm.sample(
            draws=draws,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            target_accept=0.9,
        )
        pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, random_seed=seed
        )
        pm.compute_log_likelihood(idata)

    return y_obs, idata


def _extract_pred_and_weights(
    y_obs: np.ndarray,
    idata: DataTree,
    var_name: str = "y",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull posterior-predictive draws and PSIS-LOO importance weights out of
    an ArviZ InferenceData object.

    Returns
    -------
    y_pred_flat : (n_obs, n_samples)  – predictive samples, obs on axis-0
    weights     : (n_obs, n_samples)  – normalized PSIS weights
    """
    # posterior predictive: shape (chains, draws, n_obs) → (n_obs, n_samples)
    pp = idata.posterior_predictive[var_name].values  # (chains, draws, n_obs)
    n_obs = pp.shape[2]
    y_pred_flat = pp.reshape(-1, n_obs).T  # (n_obs, n_samples)

    # PSIS-LOO weights via ArviZ
    # MAYBE JUST USE compute_psis_weights from utility
    loo_result = azs.loo(idata, pointwise=True, var_name=var_name)
    log_weights = loo_result.pareto_k.values  # we need the actual log-weights

    # ArviZ exposes the raw importance weights via psisloo wrapper
    # We recompute them from the log-likelihood stored in idata
    log_lik = idata.log_likelihood[var_name].values  # (chains, draws, n_obs)
    log_lik_flat = log_lik.reshape(-1, n_obs).T  # (n_obs, n_samples)

    # PSIS importance weights: w ∝ 1/p(y_i | θ) = exp(-log_lik)
    raw_log_w = -log_lik_flat
    # Stabilize then normalize (plain IS; replace with PSIS if you have psislw)
    log_w_stable = raw_log_w - raw_log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w_stable)
    weights = w / w.sum(axis=1, keepdims=True)

    return y_pred_flat, weights


# ============================================================================
# Tier 1: Analytical / unit tests (no PyMC)
# ============================================================================


class TestComputeLooPitAnalytical:
    """Test compute_loo_pit_model_agnostic with analytically known inputs."""

    def test_perfect_uniform_output_for_known_input(self):
        """
        If y_obs[i] is exactly the p-th quantile of the predictive samples,
        then PIT_i ≈ p.  With equal weights and enough samples this must hold.
        """
        rng = np.random.default_rng(RNG_SEED)
        n_obs, n_samples = 500, 5000
        # Draw predictive samples from N(0,1)
        y_pred = rng.standard_normal((n_obs, n_samples))
        weights = _uniform_weights(n_obs, n_samples)
        # Set y_obs to be the exact theoretical quantiles at evenly-spaced probs
        probs = np.linspace(0.01, 0.99, n_obs)
        y_obs = stats.norm.ppf(probs)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        # Each PIT value should be ≈ the corresponding prob
        np.testing.assert_allclose(pit, probs, atol=0.05)

    def test_pit_values_bounded_in_unit_interval(self):
        rng = np.random.default_rng(RNG_SEED)
        n_obs, n_samples = 100, 1000
        y_pred = rng.standard_normal((n_obs, n_samples))
        weights = _uniform_weights(n_obs, n_samples)
        y_obs = rng.standard_normal(n_obs)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        assert pit.min() >= 0.0
        assert pit.max() <= 1.0

    def test_all_predictions_below_obs_gives_pit_one(self):
        """If every predictive draw < y_obs, PIT must be 1."""
        n_obs, n_samples = 20, 200
        y_obs = np.ones(n_obs) * 10.0  # far above all predictive draws
        y_pred = np.zeros((n_obs, n_samples))
        weights = _uniform_weights(n_obs, n_samples)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        np.testing.assert_allclose(pit, np.ones(n_obs), atol=1e-10)

    def test_all_predictions_above_obs_gives_pit_zero(self):
        """If every predictive draw > y_obs, PIT must be 0."""
        n_obs, n_samples = 20, 200
        y_obs = np.ones(n_obs) * -10.0
        y_pred = np.zeros((n_obs, n_samples))
        weights = _uniform_weights(n_obs, n_samples)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        np.testing.assert_allclose(pit, np.zeros(n_obs), atol=1e-10)

    def test_weights_must_sum_to_one_per_obs(self):
        """Weights that don't sum to 1 produce meaningless PIT; catch early."""
        n_obs, n_samples = 10, 100
        y_pred = np.zeros((n_obs, n_samples))
        bad_weights = np.ones((n_obs, n_samples))  # sum = n_samples, not 1

        y_obs = np.zeros(n_obs)
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, bad_weights)
        # PIT values will be n_samples * 0.5, clearly out of [0,1]
        assert np.any(pit > 1.0), (
            "Expected PIT > 1 for un-normalized weights — "
            "the caller is responsible for normalizing."
        )

    def test_uniform_pit_passes_ks_test(self):
        """
        Simulate an oracle scenario purely in NumPy (no PyMC) and verify that
        the resulting PIT values are statistically uniform.
        """
        rng = np.random.default_rng(RNG_SEED)
        n_obs, n_samples = 500, 2000
        # True model: N(0, 1).  y_obs drawn from the same model.
        y_obs = rng.standard_normal(n_obs)
        y_pred = rng.standard_normal((n_obs, n_samples))
        weights = _uniform_weights(n_obs, n_samples)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        ks_stat, p_value = stats.kstest(pit, "uniform")
        # With n=500 we have plenty of power; p > 0.05 means we can't reject
        # uniformity — which is what we *want* for a calibrated model.
        assert p_value > 0.05, (
            f"KS test rejected uniformity for oracle NumPy PIT "
            f"(p={p_value:.4f}, stat={ks_stat:.4f})"
        )

    def test_non_uniform_pit_detected_by_ks(self):
        """
        If we deliberately shift all y_obs to be far below predictions,
        PIT values cluster near 0, and the KS test must flag this.
        """
        rng = np.random.default_rng(RNG_SEED)
        n_obs, n_samples = 300, 1000
        y_pred = rng.standard_normal((n_obs, n_samples))  # N(0,1) predictive
        y_obs = rng.normal(-3, 0.1, n_obs)  # obs << predictions → PIT near 0
        weights = _uniform_weights(n_obs, n_samples)

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, (
            f"KS test should have rejected uniformity for biased PIT (p={p_value:.4f})"
        )


# ============================================================================
# Tier 2: Oracle PyMC model (positive / calibrated tests)
# ============================================================================


@pytest.mark.slow  # mark so CI can skip long tests with -m "not slow"
class TestOracleNormalModel:
    """
    The oracle test: generate data from N(mu, sigma) and fit exactly that
    model.  Calibration must be excellent.

    These tests deliberately use a moderate n (200) and a loose tolerance
    because with n=200 some random variation is expected.
    """

    @pytest.fixture(scope="class")
    def oracle_data(self):
        """Fit once, reuse across all methods in this class."""
        y_obs, idata = _fit_oracle_normal(n=200, draws=1000, chains=2)
        y_pred_flat, weights = _extract_pred_and_weights(y_obs, idata)
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred_flat, weights)
        return {"y_obs": y_obs, "idata": idata, "pit": pit}

    def test_pit_is_approximately_uniform_ks(self, oracle_data):
        """KS p-value must be non-significant for the oracle model."""
        pit = oracle_data["pit"]
        _, p_value = stats.kstest(pit, "uniform")
        assert p_value > 0.05, (
            f"Oracle model PIT failed KS uniformity test (p={p_value:.4f}). "
            "This indicates a bug in compute_loo_pit_model_agnostic or the "
            "weight extraction."
        )

    def test_mean_pit_close_to_half(self, oracle_data):
        """E[Uniform(0,1)] = 0.5; sample mean should be within ±0.05."""
        pit = oracle_data["pit"]
        mean_pit = pit.mean()
        assert abs(mean_pit - 0.5) < 0.06, (
            f"Oracle PIT mean = {mean_pit:.3f}, expected ≈ 0.50"
        )

    def test_calibration_error_is_small(self, oracle_data):
        """
        Mean absolute calibration error (MAE between empirical and expected
        coverage) should be small for a well-specified model.
        """
        pit = oracle_data["pit"]
        expected_cov = np.arange(0.05, 1.01, 0.05)
        empirical_cov = np.array([(pit <= q).mean() for q in expected_cov])
        mae = np.mean(np.abs(empirical_cov - expected_cov))
        assert mae < 0.05, (
            f"Oracle model MAE calibration error = {mae:.4f}, expected < 0.05"
        )

    def test_few_significantly_miscalibrated_points(self, oracle_data):
        """
        With 20 coverage levels and α=0.05, we expect at most ~2 false positives
        by chance.  A well-calibrated model should rarely exceed 3.
        """
        pit = oracle_data["pit"]
        n = len(pit)
        expected_cov = np.arange(0.05, 1.01, 0.05)
        empirical_cov = np.array([(pit <= q).mean() for q in expected_cov])

        # Binomial 95% CI for each coverage level

        lower = binom.ppf(0.025, n, expected_cov) / n
        upper = binom.ppf(0.975, n, expected_cov) / n
        miscalibrated = (empirical_cov < lower) | (empirical_cov > upper)
        n_misc = int(miscalibrated.sum())

        assert n_misc <= 4, (
            f"Oracle model has {n_misc} significantly miscalibrated coverage "
            "levels (≤4 expected by chance at α=0.05 over 20 comparisons)."
        )


# ============================================================================
# Tier 3: Deliberately miscalibrated PyMC models (negative tests)
# ============================================================================


@pytest.mark.slow
class TestMiscalibratedModels:
    """
    Negative tests: verify the calibration diagnostics detect genuine
    miscalibration. Uses structural likelihood mismatches that PSIS
    cannot self-correct for.
    """

    def _run_miscalibrated(
        self,
        y_obs: np.ndarray,
        model_fn,
    ) -> np.ndarray:
        """Fit model_fn on y_obs, return PIT values."""
        idata = model_fn(y_obs)
        y_pred_flat, weights = _extract_pred_and_weights(y_obs, idata)
        return compute_loo_pit_model_agnostic(y_obs, y_pred_flat, weights)

    def test_wrong_likelihood_family_detected(self):
        """
        True DGP: heavy-tailed Student-t(nu=2).
        Fitted model: Normal → high Pareto-k values expected (PSIS warns),
        AND LOO-PIT must be non-uniform.
        """
        rng = np.random.default_rng(RNG_SEED + 10)
        n = 300
        y_obs = stats.t.rvs(df=2, loc=0, scale=1, size=n, random_state=rng)

        def fit_normal_model(y_obs):
            with pm.Model():
                mu = pm.Normal("mu", mu=0.0, sigma=10.0)
                sigma = pm.HalfNormal("sigma", sigma=5.0)
                pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
                idata = pm.sample(
                    draws=500,
                    chains=2,
                    random_seed=RNG_SEED + 10,
                    progressbar=False,
                    target_accept=0.9,
                )
                pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True, random_seed=RNG_SEED + 10
                )
                pm.compute_log_likelihood(idata)
            return idata

        # Assert the Pareto-k warning fires — it's a sign of genuine misspecification
        with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto"):
            pit = self._run_miscalibrated(y_obs, fit_normal_model)

        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, f"Wrong-likelihood model NOT detected (p={p_value:.4f})."

    def test_biased_mean_detected(self):
        """
        True DGP: N(3, 1).
        Fitted model: N(mu, 1) with mu ~ N(0, 0.1) → prior crushes mu toward 0
        → systematic bias → PIT skewed high.
        Mean shift is something PSIS cannot compensate for.
        """
        rng = np.random.default_rng(RNG_SEED + 20)
        n = 200
        y_obs = rng.normal(3.0, 1.0, size=n)

        def fit_biased_model(y_obs):
            with pm.Model():
                mu = pm.Normal("mu", mu=0.0, sigma=0.1)
                pm.Normal("y", mu=mu, sigma=1.0, observed=y_obs)
                idata = pm.sample(
                    draws=500,
                    chains=2,
                    random_seed=RNG_SEED + 20,
                    progressbar=False,
                    target_accept=0.9,
                )
                pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True, random_seed=RNG_SEED + 20
                )
                pm.compute_log_likelihood(idata)
            return idata

        pit = self._run_miscalibrated(y_obs, fit_biased_model)

        assert pit.mean() > 0.6, (
            f"Biased model: expected mean PIT > 0.6, got {pit.mean():.3f}."
        )
        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, f"Biased model not detected by KS (p={p_value:.4f})"


# # ============================================================================
# # Tier 4: Simulation-Based Calibration (SBC) – the "gold standard" approach
# # ============================================================================


# @pytest.mark.slow
# class TestSimulationBasedCalibration:
#     """
#     SBC (Talts et al., 2018): repeat the following K times —
#       1. Sample θ* ~ prior
#       2. Generate y* ~ likelihood(θ*)
#       3. Fit the model on y*
#       4. Compute rank of θ* within the posterior samples

#     For a well-specified model, ranks must be Uniform(0, S).
#     This is a model-level calibration test, complementary to LOO-PIT.
#     """

#     def test_sbc_ranks_are_uniform_normal_model(self):
#         """
#         SBC rank test for a simple N(mu, 1) model.
#         We check that the rank of the true mu is uniformly distributed.
#         """
#         K = 50  # SBC repetitions (keep low for CI speed)
#         S = 200  # posterior draws per fit
#         n_obs = 30  # observations per dataset
#         ranks = []

#         rng = np.random.default_rng(RNG_SEED)

#         for k in range(K):
#             seed_k = int(rng.integers(0, 100_000))
#             # 1. Draw from prior: mu ~ N(0, 1)
#             mu_star = rng.normal(0.0, 1.0)
#             # 2. Generate data
#             y_star = rng.normal(mu_star, 1.0, size=n_obs)

#             # 3. Fit model
#             with pm.Model():
#                 mu = pm.Normal("mu", mu=0.0, sigma=1.0)
#                 pm.Normal("y", mu=mu, sigma=1.0, observed=y_star)

#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     idata = pm.sample(
#                         draws=S,
#                         chains=1,
#                         random_seed=seed_k,
#                         progressbar=False,
#                     )

#             # 4. Rank of true mu within posterior
#             posterior_mu = idata.posterior["mu"].values.flatten()
#             rank = int((posterior_mu < mu_star).sum())
#             ranks.append(rank)

#         # KS test: ranks / S should be Uniform(0, 1)
#         _, p_value = stats.kstest(np.array(ranks) / S, "uniform")
#         assert p_value > 0.01, (
#             f"SBC rank test failed (p={p_value:.4f}). "
#             "The model posterior is not recovering the prior predictively."
#         )


# ============================================================================
# Tier 5: Integration smoke test (plot function does not crash)
# ============================================================================


@pytest.mark.slow
def test_plot_function_runs_without_error(monkeypatch):
    """
    Smoke test: plot_loo_calibration_curve_with_reference runs end-to-end
    on a known-calibrated input and returns a valid CalibrationStats object.
    """

    mpl.use("Agg")

    monkeypatch.setattr(plt, "show", lambda: None)

    rng = np.random.default_rng(RNG_SEED)
    n_obs, n_samples = 100, 2000
    y_obs = rng.standard_normal(n_obs)
    y_pred_flat = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    # Call the REAL function
    result = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs,
        y_pred=y_pred_flat,
        weights=weights,
        n_boot=500,  # small for speed
        random_seed=RNG_SEED,
    )

    # Return type
    assert isinstance(result, CalibrationStats)

    # Output shapes are consistent
    assert len(result.expected_coverage) == len(result.empirical_coverage)
    assert len(result.coverage_lower) == len(result.expected_coverage)
    assert len(result.coverage_upper) == len(result.expected_coverage)

    # Values are in valid ranges
    assert np.all(result.empirical_coverage >= 0)
    assert np.all(result.empirical_coverage <= 1)
    assert np.all(result.coverage_lower <= result.coverage_upper)
    assert 0.0 <= result.calibration_error <= 1.0
    assert result.n_miscalibrated >= 0

    # On a calibrated input, calibration error should be small
    assert result.calibration_error < 0.10, (
        f"Smoke test: expected low calibration error on oracle input, "
        f"got {result.calibration_error:.4f}"
    )


# ============================================================================
# Parametrized edge-case coverage
# ============================================================================


@pytest.mark.parametrize(
    "n_obs,n_samples",
    [
        (10, 500),  # tiny dataset
        (1000, 500),  # large dataset
        (50, 50),  # few samples
    ],
)
def test_compute_loo_pit_shapes(n_obs, n_samples):
    """Output shape is always (n_obs,) regardless of input sizes."""
    rng = np.random.default_rng(RNG_SEED)
    y_obs = rng.standard_normal(n_obs)
    y_pred = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
    assert pit.shape == (n_obs,), f"Expected ({n_obs},), got {pit.shape}"


@pytest.mark.parametrize("true_mu,true_sigma", [(0, 1), (5, 2), (-3, 0.5)])
def test_ks_uniformity_various_normals(true_mu, true_sigma):
    """
    Oracle NumPy test (no PyMC) across different N(mu, sigma) parameters.
    PIT must be uniform for any correctly-specified normal model.
    """
    rng = np.random.default_rng(RNG_SEED)
    n_obs, n_samples = 500, 3000
    y_obs = rng.normal(true_mu, true_sigma, n_obs)
    y_pred = rng.normal(true_mu, true_sigma, (n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
    _, p = stats.kstest(pit, "uniform")
    assert p > 0.05, f"KS test failed for N({true_mu}, {true_sigma}): p={p:.4f}"


@pytest.mark.visual
def test_visual_calibration_curve_oracle(monkeypatch):
    """
    Visual sanity check: plot the calibration curve for a known-calibrated
    input. The empirical coverage line should hug the 45-degree diagonal,
    with most points inside both uncertainty bands.

    Run with: pytest -m visual -s
    """

    mpl.use("TkAgg")  # interactive window — change to Qt5Agg if needed

    # Don't monkeypatch plt.show here — we WANT the window to appear
    rng = np.random.default_rng(RNG_SEED)
    # n_obs, n_samples = 300, 3000
    n_obs, n_samples = 3000, 10000
    y_obs = rng.standard_normal(n_obs)
    y_pred_flat = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    result = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs,
        y_pred=y_pred_flat,
        weights=weights,
        n_boot=2000,
        random_seed=RNG_SEED,
    )

    # Still assert programmatically so the test has a pass/fail verdict
    assert result.calibration_error < 0.05, (
        f"Oracle input should hug diagonal, got MAE={result.calibration_error:.4f}"
    )
    assert result.n_miscalibrated <= 2, (
        f"Expected ≤2 miscalibrated points, got {result.n_miscalibrated}"
    )
    print(f"\nCalibration error: {result.calibration_error:.4f}")
    print(f"Miscalibrated points: {result.n_miscalibrated}/20")
    print(">>> Close the plot window to continue <<<")
