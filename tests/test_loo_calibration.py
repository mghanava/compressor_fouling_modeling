""" Unit tests for plot_loo_calibration_curve_with_reference and
compute_loo_pit_model_agnostic.

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

from collections.abc import Callable
import io
from pathlib import Path
from typing import ClassVar, TypedDict

from arviz_base import extract, from_dict
import arviz_plots as azp
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest
from scipy import stats
from xarray import DataTree

from compressor_fouling_modeling.utility import (
    CalibrationStats,
    bayesian_bootstrap_band,
    calculate_calibration_error,
    calculate_empirical_coverage,
    calculate_miscalibrated_coverage,
    compute_loo_pit_model_agnostic,
    compute_psis_weights,
    null_coverage_band,
    plot_loo_calibration_curve_with_reference,
)

RNG_SEED = 14
# avoid shared mutable RNG module-level global to avoid consuming entropy in a
# sequential run
# RNG = np.random.default_rng(RNG_SEED)
Float64Matrix1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Float64Matrix2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Float64Matrix3D = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


class OracleData(TypedDict):
    y_obs: Float64Matrix1D
    idata: DataTree
    pit: Float64Matrix1D
    weights: Float64Matrix2D


# ============================================================================
# Helpers shared by multiple tests
# ============================================================================


def _uniform_weights(n_obs: int, n_samples: int) -> Float64Matrix2D:
    """All importance weights equal → plain empirical CDF."""
    return np.full((n_obs, n_samples), 1.0 / n_samples)


def _fit_oracle_normal(
    n: int = 200,
    true_mu: float = 0.0,
    true_sigma: float = 1.0,
    draws: int = 2000,
    chains: int = 4,
    rng_seed: int = RNG_SEED,
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
    rng = np.random.default_rng(rng_seed)
    y_obs = rng.normal(true_mu, true_sigma, size=n)

    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        _ = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

        idata = pm.sample(
            draws=draws,
            chains=chains,
            random_seed=rng,
            progressbar=False,
            target_accept=0.9,
        )
        _ = pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, random_seed=rng
        )
        _ = pm.compute_log_likelihood(idata)

    return y_obs, idata


def _extract_pred_and_weights(
    idata: DataTree,
) -> tuple[Float64Matrix2D, Float64Matrix2D]:
    """
    Pull posterior-predictive draws and PSIS-LOO importance weights out of
    an ArviZ InferenceData object.

    Returns
    -------
    y_pred_flat : (n_obs, n_samples)  -- predictive samples, obs on axis-0
    weights     : (n_obs, n_samples)  -- normalized PSIS weights
    """
    y_pred_flat: Float64Matrix2D = extract(
        idata, group="posterior_predictive", combined=True
    ).to_numpy()  # (n_obs, n_samples)

    log_lik_flat: Float64Matrix2D = extract(
        idata, group="log_likelihood", combined=True
    ).to_numpy()  # (n_obs, n_samples)
    weights, _pareto_k = compute_psis_weights(log_lik_flat)

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
    model. Calibration must be excellent.

    These tests deliberately use a moderate n (200) and a loose tolerance
    because with n=200 some random variation is expected.
    """

    @pytest.fixture
    def oracle_data(self) -> OracleData:
        """Fit once per test method."""
        y_obs, idata = _fit_oracle_normal()
        y_pred_flat, weights = _extract_pred_and_weights(idata)
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred_flat, weights)
        return {"y_obs": y_obs, "idata": idata, "pit": pit, "weights": weights}

    def test_pit_is_approximately_uniform_ks(self, oracle_data: OracleData):
        """KS p-value must be non-significant for the oracle model."""
        pit = oracle_data["pit"]
        _, p_value = stats.kstest(pit, "uniform")
        assert p_value > 0.05, (
            f"Oracle model PIT failed KS uniformity test (p={p_value:.4f}). "
            "This indicates a bug in compute_loo_pit_model_agnostic or the "
            "weight extraction."
        )

    def test_mean_pit_close_to_half(self, oracle_data: OracleData):
        """E[Uniform(0,1)] = 0.5; sample mean should be within ±0.05."""
        pit = oracle_data["pit"]
        mean_pit = pit.mean()
        assert abs(mean_pit - 0.5) < 0.05, (
            f"Oracle PIT mean = {mean_pit:.3f}, expected ≈ 0.50"
        )

    def test_calibration_error_is_small(self, oracle_data: OracleData):
        """
        Mean absolute calibration error (MAE between empirical and expected
        coverage) should be small for a well-specified model.
        """
        pit = oracle_data["pit"]
        expected_cov = np.array([*np.arange(0.05, 0.96, 0.05).tolist(), 0.99, 1.0])
        empirical_cov = calculate_empirical_coverage(pit, expected_cov)
        calibration_error, weighted_cal_error = calculate_calibration_error(
            expected_cov, empirical_cov
        )
        assert calibration_error < 0.05, (
            f"Oracle model MAE = {calibration_error:.4f}, expected < 0.05"
        )
        assert weighted_cal_error < 0.05, (
            f"Oracle model weighted MAE = {weighted_cal_error:.4f}, expected < 0.05"
        )

    def test_few_significantly_miscalibrated_points(self, oracle_data: OracleData):
        """Let X ~ Binomial(n=21, p=0.05) be the number of false positives
        (coverage levels that are flagged as significantly miscalibrated when
        the model is actually perfect). With 21 coverage levels and alpha=0.05,
        we expect E[X] = 21 * 0.05 = 1.05, so 0-2 is the typical range of false
        positives by chance. A well-calibrated model should rarely exceeds 3 as
        binom.cdf(3, 21, 0.05) = 0.98"""

        rng = np.random.default_rng(RNG_SEED)
        pit = oracle_data["pit"]
        psis_weights = oracle_data["weights"]
        expected_cov = np.array([*np.arange(0.05, 0.96, 0.05).tolist(), 0.99, 1.0])
        empirical_cov = calculate_empirical_coverage(pit, expected_cov)
        # compute finite-sample uncertainty band
        sampling_lower, sampling_upper = null_coverage_band(
            weights=psis_weights, grid=expected_cov, rng=rng
        )
        # compute posterior uncertainty
        bootstrap_lower, bootstrap_upper = bayesian_bootstrap_band(
            pit, expected_cov, rng
        )

        miscalibrated = calculate_miscalibrated_coverage(
            empirical_cov,
            expected_cov,
            sampling_lower,
            sampling_upper,
            bootstrap_lower,
            bootstrap_upper,
        )
        n_miscalibrated = np.sum(miscalibrated).astype(np.uint16)

        assert n_miscalibrated <= 3, (
            f"Oracle model has {n_miscalibrated} significantly miscalibrated coverage "
            "levels (<=3 expected by chance at alpha=0.05 over 20 comparisons)."
        )


# ============================================================================
# Tier 3: Deliberately miscalibrated PyMC models (negative tests)
# ============================================================================
@pytest.mark.slow
class TestMiscalibratedModels:
    """
    Negative tests (integration): verify LOO-PIT calibration diagnostics
    detect genuine miscalibration in fitted PyMC models.

    Each test constructs a deliberately misspecified model, fits it to data
    from a different DGP, computes LOO-PIT values via
    compute_loo_pit_model_agnostic, and checks that the resulting PIT
    distribution is detectably non-uniform.

    Unlike the unit tests in TestComputeLooPitAnalytical (which use uniform
    weights and NumPy-only inputs), these tests run real PyMC sampling and
    PSIS weight computation. The misspecifications are structural — wrong
    likelihood family or systematically biased prior — so PSIS importance
    weights cannot correct the calibration.
    """

    def _run_miscalibrated(
        self,
        y_obs: Float64Matrix1D,
        model_fn: Callable[[Float64Matrix1D], DataTree],
    ) -> Float64Matrix1D:
        """Fit model_fn on y_obs, return PIT values."""
        idata = model_fn(y_obs)
        y_pred_flat, weights = _extract_pred_and_weights(idata)
        return compute_loo_pit_model_agnostic(y_obs, y_pred_flat, weights)

    def test_wrong_likelihood_family_detected(self):
        """
        Test that LOO-PIT detects miscalibration from using the wrong
        likelihood family.

        Scenario:
          - True DGP: y_i ~ Student-t(df=2, loc=0, scale=1)
            Heavy tails produce occasional extreme observations.
          - Fitted model: Normal(mu, sigma) with weakly informative priors
            (mu ~ N(0, 10), sigma ~ HalfNormal(5)).
            A Normal likelihood cannot capture the heavy tails of a t_2.

        Consequences:
          1. PSIS warning: The Normal model treats extreme t_2 observations
             as influential, producing high Pareto-k shape parameters.
          2. PIT non-uniformity: The Normal predictive distribution is too
             narrow in the tails. Extreme t_2 observations fall far outside
             the predictive range, producing PIT values that cluster near 0
             and 1 (a U-shaped PIT histogram).

        Why PSIS cannot compensate:
          PSIS corrects for the discrepancy between the full posterior and
          the LOO posterior, but here the misspecification is a wrong
          likelihood family — the entire predictive shape is wrong. The
          importance weights are unstable (high Pareto-k), which PSIS
          already signals; the PIT values confirm the model is unusable.
        """
        rng = np.random.default_rng(RNG_SEED)
        n = 300
        y_obs = stats.t.rvs(df=2, loc=0, scale=1, size=n, random_state=rng)

        def fit_normal_model(y_obs: np.ndarray):
            with pm.Model():
                mu = pm.Normal("mu", mu=0.0, sigma=10.0)
                sigma = pm.HalfNormal("sigma", sigma=5.0)
                _ = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)
                idata = pm.sample(
                    draws=500,
                    chains=2,
                    random_seed=RNG_SEED,
                    progressbar=False,
                    target_accept=0.9,
                )
                _ = pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True, random_seed=RNG_SEED
                )
                _ = pm.compute_log_likelihood(idata)
            return idata

        # Assert the Pareto-k warning fires — it's a sign of genuine misspecification
        with pytest.warns(UserWarning, match="potential issues with LOO estimates"):
            pit = self._run_miscalibrated(y_obs, fit_normal_model)
        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, f"Wrong-likelihood model NOT detected (p={p_value:.4f})."

    def test_biased_mean_detected(self):
        """
        Test that LOO-PIT detects systematic bias from an overly strong prior
        that fights the data.

        Scenario:
          - True DGP: y_i ~ N(3, 1)  (mean = 3, sd = 1)
          - Fitted model: y_i ~ N(mu, 1) with mu ~ N(0, 0.1)
            The prior on mu has sd = 0.1 — extremely tight around 0.
            Even with n = 200 observations near 3, the posterior is heavily
            pulled toward 0. The posterior predictive therefore systematically
            underestimates every observation.

        Consequences for PIT:
          - Almost every observed y_i falls in the *right tail* of its
            predictive distribution, so PIT values cluster near 1.
          - The mean PIT is expected to exceed 0.6 (well above the
            calibrated value of 0.5).
          - A Kolmogorov-Smirnov test detects the resulting skew as
            non-uniformity (p < 0.05).

        Why PSIS cannot compensate:
          The bias is structural — the wrong prior location — not an
          overfitting artifact. PSIS corrects for the difference between
          the LOO posterior and the full posterior due to influential
          observations, but here every observation is equally miscalibrated.
          No amount of re-weighting can shift the predictive center from
          ~0 to ~3.
        """
        rng = np.random.default_rng(RNG_SEED)
        n = 200
        y_obs = rng.normal(3.0, 1.0, size=n)

        def fit_biased_model(y_obs: np.ndarray):
            with pm.Model():
                mu = pm.Normal("mu", mu=0.0, sigma=0.1)
                _ = pm.Normal("y", mu=mu, sigma=1.0, observed=y_obs)
                idata = pm.sample(
                    draws=2000,
                    chains=4,
                    random_seed=RNG_SEED + 20,
                    progressbar=False,
                    target_accept=0.9,
                )
                _ = pm.sample_posterior_predictive(
                    idata, extend_inferencedata=True, random_seed=RNG_SEED + 20
                )
                _ = pm.compute_log_likelihood(idata)
            return idata

        pit = self._run_miscalibrated(y_obs, fit_biased_model)

        assert pit.mean() > 0.6, (
            f"Biased model: expected mean PIT > 0.6, got {pit.mean():.3f}."
        )
        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, f"Biased model not detected by KS (p={p_value:.4f})"


# ============================================================================
# Tier 4: Integration smoke test (plot function does not crash)
# ============================================================================
@pytest.mark.slow
def test_plot_function_runs_without_error(
    monkeypatch: pytest.MonkeyPatch,
):
    rng = np.random.default_rng(RNG_SEED)
    """
    Smoke test: plot_loo_calibration_curve_with_reference runs end-to-end
    on a known-calibrated input and returns a valid CalibrationStats object.
    """

    mpl.use("Agg")

    monkeypatch.setattr(plt, "show", lambda: None)

    n_obs, n_samples = 100, 2000
    y_obs = rng.standard_normal(n_obs)
    y_pred_flat = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    # Call the REAL function
    _, result = plot_loo_calibration_curve_with_reference(
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
    assert len(result.bootstrap_lower) == len(result.expected_coverage)
    assert len(result.bootstrap_upper) == len(result.expected_coverage)

    # Values are in valid ranges
    assert np.all(result.empirical_coverage >= 0)
    assert np.all(result.empirical_coverage <= 1)
    assert np.all(result.bootstrap_lower <= result.bootstrap_upper)
    assert 0.0 <= result.calibration_error <= 1.0
    assert result.n_miscalibrated >= 0

    # On a calibrated input, calibration error should be small
    assert result.calibration_error < 0.10, (
        f"Smoke test: expected low calibration error on oracle input, "
        f"got {result.calibration_error:.4f}"
    )


@pytest.mark.visual
def test_visual_calibration_curve_oracle():
    rng = np.random.default_rng(RNG_SEED)
    """
    Visual sanity check: plot the calibration curve for a known-calibrated
    input. The empirical coverage line should hug the 45-degree diagonal,
    with almost all points inside both uncertainty bands.

    Run with: pytest -m visual -s
    """

    mpl.use("Agg")

    n_obs, n_samples = 1000, 10000
    y_obs = rng.standard_normal(n_obs)
    y_pred_flat = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    fig, result = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs,
        y_pred=y_pred_flat,
        weights=weights,
        random_seed=RNG_SEED,
    )
    file_path = Path(__file__).resolve()
    parent_dir = file_path.parent
    fname = Path(parent_dir / "plots" / "calibrated_model.png")
    _ = fig.figure.savefig(fname)
    # Still assert programmatically so the test has a pass/fail verdict
    assert result.calibration_error < 0.05, (
        f"Oracle input should hug diagonal, got MAE={result.calibration_error:.4f}"
    )
    assert result.n_miscalibrated <= 3, (
        f"Expected ≤2 miscalibrated points, got {result.n_miscalibrated}"
    )
    print(f"\nCalibration error: {result.calibration_error:.4f}")
    print(f"Miscalibrated points: {result.n_miscalibrated}/20")


# ============================================================================
# Tier 5: Multi-scenario visual diagnostic grid
# ============================================================================


@pytest.mark.slow
@pytest.mark.visual
class TestCalibrationScenarioGrid:
    """Multi-scenario visual calibration diagnostic grid.

    Generate data from N(0, 1) and construct 5 predictive distributions
    (calibrated, mean-shifted up/down, scale-wider/narrower). Render a
    5x6 diagnostic grid with distribution plots, PIT diagnostics, and
    LOO-PIT calibration curves.
    """

    N_OBS: int = 130
    CHAIN: int = 4
    DRAW: int = 2000
    N_SAMPLES: int = CHAIN * DRAW
    MU: float = 0.0
    SIGMA: float = 1.0
    observed: Float64Matrix1D = np.random.default_rng(RNG_SEED).normal(
        loc=MU, scale=SIGMA, size=N_OBS
    )
    predictions: ClassVar[dict[str, Float64Matrix3D]] = {}
    pits: ClassVar[dict[str, Float64Matrix1D]] = {}
    cal_errors: ClassVar[dict[str, float]] = {}

    def test_comprehensive_visual_comparison(self, monkeypatch: pytest.MonkeyPatch):
        """Orchestrate scenario generation, assertions, and grid rendering."""
        mpl.use("Agg")
        monkeypatch.setattr(plt, "show", lambda: None)

        self._generate_scenarios()
        self._assert_miscalibration_detected()
        self._build_composite_figure()

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------

    def _generate_scenarios(self):
        """Build 5 predictive distributions and compute PIT + calibration error."""
        rng = np.random.default_rng(RNG_SEED)

        for scen_mu, scen_sigma, label in [
            (self.MU + 1e-6, self.SIGMA, "calibrated"),
            (self.MU + 0.5, self.SIGMA, "mean_shift_up"),
            (self.MU - 0.5, self.SIGMA, "mean_shift_down"),
            (self.MU, self.SIGMA * 2.0, "scale_wider"),
            (self.MU, self.SIGMA * 0.5, "scale_narrower"),
        ]:
            var_name = f"y_{label}"
            pred = rng.normal(
                loc=scen_mu,
                scale=scen_sigma,
                size=(self.CHAIN, self.DRAW, self.N_OBS),
            )
            self.predictions[var_name] = pred

            y_pred_flat: Float64Matrix2D = pred.reshape(-1, self.N_OBS).T
            weights = _uniform_weights(self.N_OBS, self.N_SAMPLES)
            self.pits[label] = compute_loo_pit_model_agnostic(
                self.observed, y_pred_flat, weights
            )

            _, result = plot_loo_calibration_curve_with_reference(
                y_obs=self.observed,
                y_pred=y_pred_flat,
                weights=weights,
                n_boot=500,
                random_seed=RNG_SEED,
            )
            self.cal_errors[label] = result.calibration_error

    # ------------------------------------------------------------------
    # Programmatic assertions
    # ------------------------------------------------------------------

    def _assert_miscalibration_detected(self):
        """KS test passes for oracle, fails for miscalibrated models."""
        _, p_oracle = stats.kstest(self.pits["calibrated"], "uniform")
        assert p_oracle > 0.05, (
            f"Calibrated model PIT rejected by KS (p={p_oracle:.4f})"
        )

        for label in (
            "mean_shift_up",
            "mean_shift_down",
            "scale_wider",
            "scale_narrower",
        ):
            _, p = stats.kstest(self.pits[label], "uniform")
            assert p < 0.05, (
                f"Miscalibrated model '{label}' NOT detected by KS (p={p:.4f})"
            )

        oracle_err = self.cal_errors["calibrated"]
        assert oracle_err < min(
            v for k, v in self.cal_errors.items() if k != "calibrated"
        ), f"Oracle calibration error ({oracle_err:.4f}) should be smallest"

    # ------------------------------------------------------------------
    # Composite figure (two-pass)
    # ------------------------------------------------------------------

    def _build_composite_figure(self):
        """First pass: collect axis limits.  Second pass: render grid."""
        rng = np.random.default_rng(RNG_SEED)
        param_labels = [
            r"$(\mu + 1e-6,\ \sigma)$",
            r"$(\mu + 0.5,\ \sigma)$",
            r"$(\mu - 0.5,\ \sigma)$",
            r"$(\mu,\ 2\sigma)$",
            r"$(\mu,\ 0.5\sigma)$",
        ]

        plot_specs: list[tuple[Callable, dict]] = [  # pyright: ignore[reportMissingTypeArgument]
            (azp.plot_ppc_dist, {}),
            (azp.plot_ppc_dist, {"kind": "ecdf"}),
            (
                azp.plot_ppc_pit,
                {
                    "envelope_prob": 0.95,
                    "stats": {"ecdf_pit": {"n_simulations": 3000, "rng": rng}},
                },
            ),
            (
                azp.plot_ppc_pit,
                {
                    "coverage": True,
                    "envelope_prob": 0.95,
                    "stats": {"ecdf_pit": {"n_simulations": 3000, "rng": rng}},
                },
            ),
        ]

        dt_i = from_dict(
            {
                "posterior_predictive": self.predictions,
                "observed_data": dict.fromkeys(self.predictions, self.observed),
            }
        )
        var_names = list(self.predictions.keys())

        x_lim_global, y_lim_global = self._collect_limits(dt_i, var_names, plot_specs)
        self._render_grid(
            dt_i, var_names, plot_specs, x_lim_global, y_lim_global, param_labels
        )

    def _collect_limits(
        self,
        dt_i: DataTree,
        var_names: list[str],
        plot_specs: list[tuple[Callable, dict]],  # pyright: ignore[reportMissingTypeArgument]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """First pass: render plots, record axis limits, close each figure."""
        x_limits_col12: list[tuple[float, float]] = []
        y_limits_col34: list[tuple[float, float]] = []

        for var_name in var_names:
            for col, (plot_fn, plot_kwargs) in enumerate(plot_specs):
                pc = plot_fn(
                    dt_i,
                    var_names=[var_name],
                    group="posterior_predictive",
                    figure_kwargs={"figsize": (3.5, 3.5)},
                    **plot_kwargs,
                )
                fig: Figure = pc.viz.ds["figure"].to_numpy().item()
                ax = fig.axes[0]
                if col < 2:
                    x_limits_col12.append(ax.get_xlim())
                else:
                    y_limits_col34.append(ax.get_ylim())
                plt.close(fig)

        x_abs = max(abs(v) for lim in x_limits_col12 for v in lim)
        y_abs = max(abs(v) for lim in y_limits_col34 for v in lim)
        return (-x_abs, x_abs), (-y_abs, y_abs)

    def _render_grid(
        self,
        dt_i: DataTree,
        var_names: list[str],
        plot_specs: list[tuple[Callable, dict]],  # pyright: ignore[reportMissingTypeArgument]
        x_lim_global: tuple[float, float],
        y_lim_global: tuple[float, float],
        param_labels: list[str],
    ):
        """Second pass: compose 5x6 grid with unified axis limits."""
        plt.close("all")
        fig = plt.figure(figsize=(30, 30))
        gs = GridSpec(5, 6, figure=fig, width_ratios=[0.8, 1, 1, 1, 1, 1.6])
        _ = fig.suptitle(
            rf"$\mu = {self.MU},\ \sigma = {self.SIGMA}$", fontsize=16, x=0.35
        )

        for row, var_name in enumerate(var_names):
            self._render_label_cell(fig, gs, row, param_labels[row])
            self._render_arviz_cells(
                fig, gs, row, var_name, dt_i, plot_specs, x_lim_global, y_lim_global
            )
            self._render_cal_curve_cell(fig, gs, row, var_name)

        plt.tight_layout()
        fname = (
            Path(__file__).resolve().parent
            / "plots"
            / "calibration_diagnostic_grid.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close("all")

    @staticmethod
    def _render_label_cell(fig: Figure, gs: GridSpec, row: int, label: str):
        """Render a single label cell (column 0 of the grid)."""
        ax_label = fig.add_subplot(gs[row, 0])
        _ = ax_label.text(
            0.5,
            0.5,
            label,
            transform=ax_label.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
            c="blue",
        )
        _ = ax_label.axis("off")

    def _render_arviz_cells(
        self,
        fig: Figure,
        gs: GridSpec,
        row: int,
        var_name: str,
        dt_i: DataTree,
        plot_specs: list[tuple[Callable, dict]],  # pyright: ignore[reportMissingTypeArgument]
        x_lim_global: tuple[float, float],
        y_lim_global: tuple[float, float],
    ):
        """Render the 4 arviz-based plot cells (columns 1-4) for one row."""
        for col in range(4):
            ax = fig.add_subplot(gs[row, col + 1])
            plot_fn, plot_kwargs = plot_specs[col]
            pc = plot_fn(
                dt_i,
                var_names=[var_name],
                group="posterior_predictive",
                figure_kwargs={"figsize": (3.5, 3.5)},
                **plot_kwargs,
            )
            plot_fig: Figure = pc.viz.ds["figure"].to_numpy().item()
            xlim = x_lim_global if col < 2 else None
            ylim = y_lim_global if col >= 2 else None
            img = self._render_plot_to_img(plot_fig, xlim, ylim)
            _ = ax.imshow(img)
            _ = ax.axis("off")

    def _render_cal_curve_cell(
        self, fig: Figure, gs: GridSpec, row: int, var_name: str
    ):
        """Render the LOO-PIT calibration curve (column 5) for one row."""
        ax_cal = fig.add_subplot(gs[row, 5])
        y_pred_var: Float64Matrix2D = (
            self.predictions[var_name].reshape(-1, self.N_OBS).T
        )
        var_weights = _uniform_weights(self.N_OBS, self.N_SAMPLES)
        _ = plot_loo_calibration_curve_with_reference(
            y_obs=self.observed,
            y_pred=y_pred_var,
            weights=var_weights,
            n_boot=10000,
            figsize=(3.5, 3.5),
            random_seed=RNG_SEED,
            ax=ax_cal,
        )

    @staticmethod
    def _render_plot_to_img(
        fig: Figure,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Save figure to PNG buffer, close it, return image array."""
        if xlim is not None:
            _ = fig.axes[0].set_xlim(xlim)
        if ylim is not None:
            _ = fig.axes[0].set_ylim(ylim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        _ = buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return img


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
def test_compute_loo_pit_shapes(
    n_obs: int,
    n_samples: int,
):
    """Output shape is always (n_obs,) regardless of input sizes."""
    rng = np.random.default_rng(RNG_SEED)
    y_obs = rng.standard_normal(n_obs)
    y_pred = rng.standard_normal((n_obs, n_samples))
    weights = _uniform_weights(n_obs, n_samples)

    pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
    assert pit.shape == (n_obs,), f"Expected ({n_obs},), got {pit.shape}"


@pytest.mark.parametrize("true_mu,true_sigma", [(0, 1), (5, 2), (-3, 0.5)])
def test_ks_uniformity_various_normals(
    true_mu: float,
    true_sigma: float,
):
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
