"""Unit tests for plot_loo_calibration_curve_with_reference and related helpers.

Test strategy
-------------
Five tiers of tests:

1. **Analytical / unit tests** - feed known inputs (pre-computed PIT values,
   exact uniform samples) directly into ``compute_loo_pit_model_agnostic``
   and assert outputs are correct without touching PyMC.

1b. **Calibration curve helper unit tests** - test ``null_coverage_band``,
   ``bayesian_bootstrap_band``, ``calculate_empirical_coverage``,
   ``calculate_calibration_error``, and ``calculate_miscalibrated_coverage``
   with synthetic inputs to verify invariants (bounds, symmetry,
   monotonicity, sensitivity to ESS/n, edge cases).

2. **Oracle (self-consistent) PyMC model** - generate data *from* a model,
   then fit *that exact model* on it.  Because the model is correctly
   specified, LOO-PIT must be approximately Uniform(0,1) and the calibration
   curve must lie on the diagonal.  This is the gold-standard positive test.

3. **Deliberately miscalibrated PyMC models** - fit a model that is *wrong*
   (e.g. variance too small, wrong likelihood family, biased prior).
   These are negative tests: the diagnostic *must* detect miscalibration.

4. **Integration smoke test** - ``plot_loo_calibration_curve_with_reference``
   runs end-to-end on a known-calibrated input and returns valid
   ``CalibrationStats``.

5. **Multi-scenario visual diagnostic grid** - generates a 5x6 diagnostic
   grid covering calibrated and miscalibrated scenarios (mean-shifted up/down,
   scale wider/narrower) and asserts the KS test passes/fails accordingly.
"""

from __future__ import annotations

from collections.abc import Callable
import io
import math
from pathlib import Path
from typing import ClassVar, TypedDict, cast, override
import unittest

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
from xarray import DataArray, DataTree

from compressor_fouling_modeling.utility import (
    CalibrationStats,
    Float64Matrix1D,
    Float64Matrix3D,
    _resolve_likelihood_var_name,  # pyright: ignore[reportPrivateUsage] # ruff: ignore[import-private-name]
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
N_OBS: int = 500
N_CHAIN: int = 4
N_DRAW: int = 1000


class OracleData(TypedDict):
    pit: DataArray
    weights: DataArray


# ============================================================================
# Helpers shared by multiple tests
# ============================================================================


def _uniform_weights(n_obs: int, n_chain: int, n_draw: int) -> Float64Matrix3D:
    """All importance weights equal → plain empirical CDF."""
    n_samples = n_chain * n_draw
    return np.full((n_obs, n_chain, n_draw), 1.0 / n_samples)


def _make_pit_inputs(
    y_obs_values: Float64Matrix1D,
    y_pred_values: Float64Matrix3D,
    weights_values: Float64Matrix3D,
) -> tuple[DataArray, DataArray, DataArray]:
    """Wrap raw numpy arrays as DataArrays sharing one obs coord, for tests
    exercising compute_loo_pit_model_agnostic (which requires named dims and
    validates obs alignment via xr.align)."""
    n_obs = y_obs_values.shape[0]
    obs = np.arange(n_obs)
    y_obs = DataArray(y_obs_values, dims="obs", coords={"obs": obs})
    y_pred = DataArray(
        y_pred_values, dims=("chain", "draw", "obs"), coords={"obs": obs}
    )
    weights = DataArray(
        weights_values, dims=("obs", "chain", "draw"), coords={"obs": obs}
    )
    return y_obs, y_pred, weights


def _convert_weights_numpy_to_dataarray(weights_values: Float64Matrix3D) -> DataArray:
    return DataArray(weights_values, dims=("obs", "chain", "draw"))


def _convert_grid_numpy_to_dataarray(
    values: Float64Matrix1D, grid: Float64Matrix1D
) -> DataArray:
    """Wrap a numpy array as a DataArray living on a shared `grid` coordinate.

    Use this for any test fixture that will be passed to a function expecting
    `grid`-dimensioned DataArrays (e.g. calculate_calibration_error,
    calculate_miscalibrated_coverage, null_coverage_band,
    bayesian_bootstrap_band).

    `values` and `grid` are usually different arrays: `values` is the data at
    each point (e.g. an empirical coverage curve, a band's lower bound), while
    `grid` is the shared x-axis position each point is *at* (typically the
    test's expected_coverage array). Passing the same array as both is correct
    only when you are building the base grid axis itself:

        expected_coverage = np.linspace(0.05, 0.95, 19)
        exp = _convert_grid_numpy_to_dataarray(expected_coverage, expected_coverage)

    For everything else, pass that same `expected_coverage` array as `grid` and
    the function's own output as `values`, so every DataArray in the test shares
    one `grid` coordinate -- this is what lets `xr.align(join="exact")` inside
    the function under test pass instead of raising, and is required for
    correctness even when two arrays happen to be the same length:

        emp = _convert_grid_numpy_to_dataarray(empirical_values, expected_coverage)
        sl  = _convert_grid_numpy_to_dataarray(sampling_lower_values, expected_coverage)

    Args:
        values: The data to store at each grid point. Shape (m,).
        grid: The shared grid coordinate values (the x-axis). Shape (m,).
            Must be the *same* array/values across every DataArray you intend
            to combine, align, or compare in the same call -- using each
            array's own `values` as its own `grid` (i.e. calling this with
            `values is grid` for a non-base-axis array) silently produces
            mismatched coordinates and will make `xr.align(join="exact")`
            raise inside the function under test.

    Returns:
        DataArray with dim "grid", holding `values` as data and `grid` as
        the coordinate.
    """
    return DataArray(values, dims="grid", coords={"grid": grid})


def _fit_oracle_normal(
    n: int = 200,
    true_mu: float = 0.0,
    true_sigma: float = 1.0,
    draws: int = 2000,
    chains: int = 4,
    rng_seed: int = RNG_SEED,
) -> tuple[DataTree, DataArray]:
    """
    Oracle normal model: data ~ N(mu, sigma).

    We generate data from the exact likelihood and fit the same model, so the
    posterior predictive *must* be calibrated.

    Returns
        An idata which is ArviZ InferenceData with posterior_predictive and
        log_likelihood.
    """
    rng = np.random.default_rng(rng_seed)
    y_obs = rng.normal(true_mu, true_sigma, size=n)

    with pm.Model(coords={"obs": np.arange(len(y_obs))}):
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        _ = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")

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

    var_name = _resolve_likelihood_var_name(idata)
    y_obs = idata.observed_data[var_name]
    return idata, y_obs


def _extract_pred_and_weights(
    idata: DataTree,
) -> tuple[DataArray, DataArray]:
    """
    Pull posterior-predictive draws and PSIS-LOO importance weights out of
    an ArviZ InferenceData object.

    Returns
        A tuple (y_pred, weights), where y_pred has dims (chain, draw, obs) and
        weights has dims (obs, chain, draw), matching what
        compute_loo_pit_model_agnostic requires.
    """
    y_pred = cast(
        DataArray, extract(idata, group="posterior_predictive", combined=False)
    )  # (chain, draw, obs)

    weights = compute_psis_weights(idata)[2]

    return y_pred, weights


# ============================================================================
# Tier 1a: Analytical / unit tests (no PyMC)
# ============================================================================


class TestComputeLooPitAnalytical(unittest.TestCase):
    """Test compute_loo_pit_model_agnostic with analytically known inputs."""

    rng: np.random.Generator  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self):
        """Correctly overrides unittest.TestCase.setUp"""
        self.rng = np.random.default_rng(RNG_SEED)

    def test_perfect_uniform_output_for_known_input(self):
        """
        If y_obs[i] is exactly the p-th quantile of the predictive samples,
        then PIT_i ≈ p.  With equal weights and enough samples this must hold.
        """
        probs = np.linspace(0.01, 0.99, N_OBS)

        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=stats.norm.ppf(
                probs
            ),  # set y_obs to be the exact theoretical quantiles at evenly-spaced probs
            y_pred_values=self.rng.standard_normal(
                (N_CHAIN, N_DRAW, N_OBS)
            ),  # draw predictive samples from N(0,1)
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )

        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        # Each PIT value should be ≈ the corresponding prob
        np.testing.assert_allclose(pit, probs, atol=0.05)

    def test_pit_values_bounded_in_unit_interval(self):

        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=self.rng.standard_normal(N_OBS),
            y_pred_values=self.rng.standard_normal((N_CHAIN, N_DRAW, N_OBS)),
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        assert pit.min() >= 0.0
        assert pit.max() <= 1.0

    def test_all_predictions_below_obs_gives_pit_one(self):
        """If every predictive draw < y_obs, PIT must be 1."""

        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=np.ones(N_OBS) * 10.0,  # far above all predictive draws,
            y_pred_values=np.zeros((N_CHAIN, N_DRAW, N_OBS)),
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        np.testing.assert_allclose(pit, np.ones(N_OBS), atol=1e-10)

    def test_all_predictions_above_obs_gives_pit_zero(self):
        """If every predictive draw > y_obs, PIT must be 0."""

        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=np.ones(N_OBS) * -10.0,  # far below all predictive draws,
            y_pred_values=np.zeros((N_CHAIN, N_DRAW, N_OBS)),
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        np.testing.assert_allclose(pit, np.zeros(N_OBS), atol=1e-10)

    def test_uniform_pit_passes_ks_test(self):
        """
        Simulate an oracle scenario purely in NumPy (no PyMC) and verify that
        the resulting PIT values are statistically uniform.
        """
        # True model: N(0, 1). y_obs drawn from the same model.
        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=self.rng.standard_normal(N_OBS),
            y_pred_values=self.rng.standard_normal((N_CHAIN, N_DRAW, N_OBS)),
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )
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
        y_obs, y_pred, weights = _make_pit_inputs(
            y_obs_values=self.rng.normal(
                -3, 0.1, N_OBS
            ),  # obs << predictions → PIT near 0
            y_pred_values=self.rng.standard_normal(
                (N_CHAIN, N_DRAW, N_OBS)
            ),  # N(0,1) predictive
            weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
        )
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

        _, p_value = stats.kstest(pit, "uniform")
        assert p_value < 0.05, (
            f"KS test should have rejected uniformity for biased PIT (p={p_value:.4f})"
        )


# ============================================================================
# Tier 1b: Calibration curve helper unit tests (no PyMC)
# ============================================================================


class TestNullCoverageBand(unittest.TestCase):
    """Unit tests for null_coverage_band."""

    rng: np.random.Generator  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self):
        """Correctly overrides unittest.TestCase.setUp"""
        self.rng = np.random.default_rng(RNG_SEED)

    def test_bounds_ordered(self):
        """Lower bound must be ≤ upper bound at every grid point."""
        grid_values = np.linspace(0.05, 0.95, 19)
        lower, upper = null_coverage_band(
            _convert_weights_numpy_to_dataarray(
                _uniform_weights(N_OBS, N_CHAIN, N_DRAW)
            ),
            _convert_grid_numpy_to_dataarray(grid_values, grid_values),
            self.rng,
            B=500,
        )
        assert np.all(lower <= upper + 1e-15), "lower > upper at some grid point"

    def test_bounds_in_unit_interval(self):
        """Band bounds must lie in [0, 1] as grid lies in [0, 1]."""
        grid_values = np.linspace(0.05, 0.95, 19)
        lower, upper = null_coverage_band(
            _convert_weights_numpy_to_dataarray(
                _uniform_weights(N_OBS, N_CHAIN, N_DRAW)
            ),
            _convert_grid_numpy_to_dataarray(grid_values, grid_values),
            self.rng,
            B=500,
        )
        assert np.all(lower >= 0.0), f"lower bound below 0: {lower}"
        assert np.all(upper <= 1.0 + 1e-15), f"upper bound above 1: {upper}"

    def test_band_centers_on_diagonal(self):
        """For uniform weights, band should be centered near the diagonal."""
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        lower, upper = null_coverage_band(
            _convert_weights_numpy_to_dataarray(
                _uniform_weights(N_OBS, N_CHAIN, N_DRAW)
            ),
            grid,
            self.rng,
            B=1000,
        )
        midpoint = (lower + upper) / 2
        assert np.all(np.abs(midpoint - grid) < 0.05), (
            f"Band midpoint {midpoint} deviates from grid {grid}"
        )

    def test_band_narrows_with_more_observations(self):
        """Band width should decrease as n_obs increases (higher ESS)."""
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        lo20, up20 = null_coverage_band(
            _convert_weights_numpy_to_dataarray(_uniform_weights(20, N_CHAIN, N_DRAW)),
            grid,
            self.rng,
            B=500,
        )
        lo200, up200 = null_coverage_band(
            _convert_weights_numpy_to_dataarray(_uniform_weights(200, N_CHAIN, N_DRAW)),
            grid,
            self.rng,
            B=500,
        )
        assert (up20 - lo20)[0] > (up200 - lo200)[0], (
            "Larger n_obs should produce narrower band"
        )

    def test_low_ess_produces_wider_band_than_high_ess(self):
        """Near-degenerate weights (ESS ≈ 1) should produce a wider null band
        than well-mixed weights (ESS ≈ n_samples) at the same observation count."""
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)

        # Low ESS: all mass on a single (chain, draw) per observation → ESS ≈ 1
        low_ess_weights = np.zeros((N_OBS, N_CHAIN, N_DRAW))
        low_ess_weights[:, 0, 0] = 1.0

        # High ESS: uniform weights across chain/draw → ESS ≈ N_CHAIN * N_DRAW
        high_ess_weights = np.ones((N_OBS, N_CHAIN, N_DRAW))

        # Use independent RNGs (or copies of the same seeded state) so the two
        # calls aren't confounded by consuming the shared rng differently.
        rng_low = np.random.default_rng(RNG_SEED)
        rng_high = np.random.default_rng(RNG_SEED)

        lower_low, upper_low = null_coverage_band(
            _convert_weights_numpy_to_dataarray(low_ess_weights),
            grid,
            rng_low,
            B=500,
        )
        lower_high, upper_high = null_coverage_band(
            _convert_weights_numpy_to_dataarray(high_ess_weights),
            grid,
            rng_high,
            B=500,
        )

        width_low = upper_low - lower_low
        width_high = upper_high - lower_high
        # Use an aggregate statistic instead of .all() across grid points.
        # Comparing at every single grid point independently multiplies the
        # chances of a noisy flip at any one point
        mean_width_low = float(width_low.mean())
        mean_width_high = float(width_high.mean())

        assert mean_width_low > mean_width_high, (
            f"Expected low-ESS band wider on average: "
            f"low={mean_width_low:.3f}, high={mean_width_high:.3f}"
        )

    def test_reproducible_with_same_seed(self):
        """Same RNG seed must produce identical band."""
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        weights_da = _convert_weights_numpy_to_dataarray(
            _uniform_weights(N_OBS, N_CHAIN, N_DRAW)
        )
        # Each call gets its own fresh Generator, both starting at the identical
        # seed state S0. Since both start at S0 and (presumably) consume the
        # same sequence of draws in the same order, they produce identical
        # output. This is genuinely testing "same seed → same result".
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        lower1, upper1 = null_coverage_band(weights_da, grid, rng1, B=500)
        lower2, upper2 = null_coverage_band(weights_da, grid, rng2, B=500)
        np.testing.assert_array_equal(lower1, lower2)
        np.testing.assert_array_equal(upper1, upper2)


class TestBayesianBootstrapBand(unittest.TestCase):
    """Unit tests for bayesian_bootstrap_band."""

    rng: np.random.Generator  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self):
        """Correctly overrides unittest.TestCase.setUp"""
        self.rng = np.random.default_rng(RNG_SEED)

    def test_bounds_ordered(self):
        """Lower bound must be ≤ upper bound at every grid point."""
        loo_pit = DataArray(self.rng.uniform(0, 1, size=N_OBS), dims="obs")
        grid_values = np.linspace(0.05, 0.95, 19)
        lower, upper = bayesian_bootstrap_band(
            loo_pit,
            _convert_grid_numpy_to_dataarray(grid_values, grid_values),
            self.rng,
            B=500,
        )
        assert np.all(lower <= upper + 1e-15), "lower > upper at some grid point"

    def test_bounds_in_unit_interval(self):
        """Band bounds must lie in [0, 1] as grid lies in [0, 1]."""
        loo_pit = DataArray(self.rng.uniform(0, 1, size=N_OBS), dims="obs")
        grid_values = np.linspace(0.05, 0.95, 19)
        lower, upper = bayesian_bootstrap_band(
            loo_pit,
            _convert_grid_numpy_to_dataarray(grid_values, grid_values),
            self.rng,
            B=500,
        )
        assert np.all(lower >= 0.0), f"lower bound below 0: {lower}"
        assert np.all(upper <= 1.0 + 1e-15), f"upper bound above 1: {upper}"

    def test_band_symmetric_for_uniform_pit(self):
        """For uniform PIT, band should be symmetric around the diagonal."""
        loo_pit = DataArray(self.rng.uniform(0, 1, size=N_OBS), dims="obs")
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        lower, upper = bayesian_bootstrap_band(loo_pit, grid, self.rng, B=1000)
        midpoint = (lower + upper) / 2
        assert np.all(np.abs(midpoint - grid) < 0.03), (
            f"Midpoint {midpoint} deviates from grid {grid}"
        )

    def test_band_narrows_with_more_observations(self):
        """Band width should decrease as n increases."""
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        pit30 = DataArray(
            np.random.default_rng(RNG_SEED).uniform(0, 1, size=30), dims="obs"
        )
        pit300 = DataArray(self.rng.uniform(0, 1, size=300), dims="obs")
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        lo30, up30 = bayesian_bootstrap_band(pit30, grid, rng1, B=500)
        lo300, up300 = bayesian_bootstrap_band(pit300, grid, rng2, B=500)
        assert (up30 - lo30)[0] > (up300 - lo300)[0], (
            f"Expected narrower band for larger n, got {(up30 - lo30)[0]:.4f} → {(up300 - lo300)[0]:.4f}"
        )

    def test_band_snaps_to_zero_one_at_edges(self):
        """At grid=0 and grid=1 the band should be essentially 0 and 1."""
        loo_pit = DataArray(self.rng.uniform(0, 1, size=N_OBS), dims="obs")
        grid_values = np.array([0.0, 1.0])
        lower, upper = bayesian_bootstrap_band(
            loo_pit,
            _convert_grid_numpy_to_dataarray(grid_values, grid_values),
            self.rng,
            B=500,
        )
        np.testing.assert_allclose(lower[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(upper[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(lower[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(upper[1], 1.0, atol=1e-10)

    def test_reproducible_with_same_seed(self):
        """Same RNG seed must produce identical band."""
        rng = np.random.default_rng(RNG_SEED)
        n = 100
        loo_pit = DataArray(rng.uniform(0, 1, size=n), dims="obs")
        grid_values = np.array([0.25, 0.50, 0.75])
        grid = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        rng1 = np.random.default_rng(RNG_SEED)
        rng2 = np.random.default_rng(RNG_SEED)
        lower1, upper1 = bayesian_bootstrap_band(loo_pit, grid, rng1, B=500)
        lower2, upper2 = bayesian_bootstrap_band(loo_pit, grid, rng2, B=500)
        np.testing.assert_array_equal(lower1, lower2)
        np.testing.assert_array_equal(upper1, upper2)


class TestCalculateEmpiricalCoverage:
    """Unit tests for calculate_empirical_coverage."""

    def test_perfect_uniform_pit(self):
        """Uniform PIT should give coverage ≈ expected level."""
        rng = np.random.default_rng(RNG_SEED)
        n = 10000
        loo_pit = DataArray(rng.uniform(0, 1, size=n), dims="obs")
        grid_values = np.array([0.25, 0.50, 0.75])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        emp = calculate_empirical_coverage(loo_pit, expected)
        assert np.all(np.abs(emp - expected) < 0.01), (
            f"Empirical coverage {emp} deviates from expected {expected}"
        )

    def test_all_below_threshold(self):
        """If all PIT values are below the smallest threshold, coverage = 1."""
        loo_pit = DataArray(np.array([0.01, 0.02, 0.03]), dims="obs")
        grid_values = np.array([0.05, 0.10])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        emp = calculate_empirical_coverage(loo_pit, expected)
        np.testing.assert_array_equal(emp, np.array([1.0, 1.0]))

    def test_all_above_threshold(self):
        """If all PIT values are above the largest threshold, coverage = 0."""
        loo_pit = DataArray(np.array([0.9, 0.95, 0.99]), dims="obs")
        grid_values = np.array([0.25, 0.50])
        emp = calculate_empirical_coverage(
            loo_pit, _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        )
        np.testing.assert_array_equal(emp, np.array([0.0, 0.0]))

    def test_monotonic_increasing(self):
        """Empirical coverage must be non-decreasing with expected coverage."""
        rng = np.random.default_rng(RNG_SEED)
        loo_pit = DataArray(rng.uniform(0, 1, size=1000), dims="obs")
        grid_values = np.linspace(0.05, 0.95, 19)
        emp = calculate_empirical_coverage(
            loo_pit, _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        )
        assert np.all(np.diff(emp) >= -1e-15), (
            "Empirical coverage decreased for higher expected level"
        )


class TestCalculateCalibrationError:
    """Unit tests for calculate_calibration_error."""

    def test_perfect_calibration_zero_error(self):
        """Perfect calibration yields error of 0."""
        grid_values = np.array([0.25, 0.50, 0.75])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        empirical = expected.copy()
        err, werr = calculate_calibration_error(expected, empirical)
        assert math.isclose(err, 0.0)
        assert math.isclose(werr, 0.0)

    def test_known_deviation(self):
        """A known constant offset produces the expected MAE."""
        grid_values = np.array([0.25, 0.50, 0.75])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        offset = 0.05
        empirical = expected + offset
        err, _werr = calculate_calibration_error(expected, empirical)
        assert abs(err - offset) < 1e-15

    def test_weighted_error_differs_from_unweighted(self):
        """Weighted error should differ when deviations are concentrated in mid-range."""
        grid_values = np.array([0.1, 0.5, 0.9])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        empirical = _convert_grid_numpy_to_dataarray(
            np.array([0.3, 0.3, 0.7]), grid_values
        )
        err, werr = calculate_calibration_error(expected, empirical)
        assert err != werr, f"Weighted error {werr} should differ from unweighted {err}"

    def test_all_identical_arrays(self):
        """Identical arrays produce zero error regardless of values."""
        grid_values = np.array([0.0, 0.3, 0.7, 1.0])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        empirical = expected.copy()
        err, werr = calculate_calibration_error(expected, empirical)
        assert math.isclose(err, 0.0)
        assert math.isclose(werr, 0.0)

    def test_weighted_error_is_nonnegative(self):
        """Both error metrics must be non-negative."""
        grid_values = np.array([0.25, 0.50, 0.75])
        expected = _convert_grid_numpy_to_dataarray(grid_values, grid_values)
        empirical = _convert_grid_numpy_to_dataarray(
            np.array([0.30, 0.45, 0.80]), grid_values
        )
        err, werr = calculate_calibration_error(expected, empirical)
        assert err >= 0.0
        assert werr >= 0.0


class TestCalculateMiscalibratedCoverage:
    """Unit tests for calculate_miscalibrated_coverage."""

    def test_all_overlapping_no_miscalibration(self):
        """When sampling and bootstrap bands overlap at all levels, none flagged."""
        exp_values = np.array([0.25, 0.50, 0.75])

        exp = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
        emp = _convert_grid_numpy_to_dataarray(np.array([0.25, 0.50, 0.75]), exp_values)
        sl = _convert_grid_numpy_to_dataarray(np.array([0.20, 0.45, 0.70]), exp_values)
        su = _convert_grid_numpy_to_dataarray(np.array([0.30, 0.55, 0.80]), exp_values)
        bl = _convert_grid_numpy_to_dataarray(np.array([0.22, 0.47, 0.72]), exp_values)
        bu = _convert_grid_numpy_to_dataarray(np.array([0.28, 0.53, 0.78]), exp_values)
        flagged = calculate_miscalibrated_coverage(emp, exp, sl, su, bl, bu)
        assert not np.any(flagged), f"No levels should be miscalibrated: {flagged}"

    def test_all_non_overlapping_all_miscalibrated(self):
        """When bands do not overlap and emp ≠ exp, all flagged."""
        exp_values = np.array([0.25, 0.50, 0.75])

        exp = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
        emp = _convert_grid_numpy_to_dataarray(np.array([0.9, 0.9, 0.9]), exp_values)
        # sampling band sits low, bootstrap band sits high → no overlap
        sl = _convert_grid_numpy_to_dataarray(np.array([0.05, 0.05, 0.05]), exp_values)
        su = _convert_grid_numpy_to_dataarray(np.array([0.15, 0.15, 0.15]), exp_values)
        bl = _convert_grid_numpy_to_dataarray(np.array([0.80, 0.80, 0.80]), exp_values)
        bu = _convert_grid_numpy_to_dataarray(np.array([0.95, 0.95, 0.95]), exp_values)
        flagged = calculate_miscalibrated_coverage(emp, exp, sl, su, bl, bu)
        assert np.all(flagged), f"All levels should be miscalibrated: {flagged}"

    def test_partial_overlap(self):
        """Only levels where bands do not overlap and emp deviates should flag."""
        exp_values = np.array([0.25, 0.50, 0.75])

        exp = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
        emp = _convert_grid_numpy_to_dataarray(np.array([0.25, 0.90, 0.75]), exp_values)
        # index 0: emp≈exp, bands overlap → calibrated
        # index 1: emp>>exp, bands don't overlap → miscalibrated
        # index 2: emp≈exp, bands overlap → calibrated
        sl = _convert_grid_numpy_to_dataarray(np.array([0.20, 0.10, 0.70]), exp_values)
        su = _convert_grid_numpy_to_dataarray(np.array([0.30, 0.20, 0.80]), exp_values)
        bl = _convert_grid_numpy_to_dataarray(np.array([0.22, 0.80, 0.72]), exp_values)
        bu = _convert_grid_numpy_to_dataarray(np.array([0.28, 0.90, 0.78]), exp_values)
        flagged = calculate_miscalibrated_coverage(emp, exp, sl, su, bl, bu)
        expected_flag = np.array([False, True, False])
        np.testing.assert_array_equal(flagged, expected_flag)

    def test_sampling_band_encompasses_bootstrap(self):
        """If sampling band fully contains bootstrap band → all calibrated."""
        exp_values = np.array([0.25, 0.50, 0.75])

        exp = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
        emp = _convert_grid_numpy_to_dataarray(np.array([0.25, 0.50, 0.75]), exp_values)
        sl = _convert_grid_numpy_to_dataarray(
            np.array([0.0, 0.0, 0.0]), exp_values
        )  # very wide sampling band
        su = _convert_grid_numpy_to_dataarray(np.array([1.0, 1.0, 1.0]), exp_values)
        bl = _convert_grid_numpy_to_dataarray(np.array([0.22, 0.47, 0.72]), exp_values)
        bu = _convert_grid_numpy_to_dataarray(np.array([0.28, 0.53, 0.78]), exp_values)
        flagged = calculate_miscalibrated_coverage(emp, exp, sl, su, bl, bu)
        assert not np.any(flagged), (
            "All levels should be calibrated when sampling band contains bootstrap band"
        )

    def test_empirical_far_below_expected(self):
        """When empirical far below expected and bands do not overlap, flag all."""
        exp_values = np.array([0.25, 0.50, 0.75])

        exp = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
        emp = _convert_grid_numpy_to_dataarray(np.array([0.05, 0.10, 0.20]), exp_values)
        # sampling band sits entirely above bootstrap band → no overlap
        sl = _convert_grid_numpy_to_dataarray(np.array([0.30, 0.55, 0.80]), exp_values)
        su = _convert_grid_numpy_to_dataarray(np.array([0.35, 0.60, 0.85]), exp_values)
        bl = _convert_grid_numpy_to_dataarray(np.array([0.02, 0.05, 0.10]), exp_values)
        bu = _convert_grid_numpy_to_dataarray(np.array([0.08, 0.15, 0.25]), exp_values)
        flagged = calculate_miscalibrated_coverage(emp, exp, sl, su, bl, bu)
        assert np.all(flagged), (
            "Empirical far from expected should be flagged everywhere"
        )


# ============================================================================
# Tier 1c: Parametrized edge-case coverage (no PyMC)
# ============================================================================


@pytest.mark.parametrize(
    "n_obs,n_chain, n_draw",
    [
        (10, 1, 500),  # tiny dataset
        (1000, 1, 500),  # large dataset
        (50, 1, 50),  # few samples
    ],
)
def test_compute_loo_pit_shapes(
    n_obs: int,
    n_chain: int,
    n_draw: int,
):
    """Output shape is always (n_obs,) regardless of input sizes.

    ``@pytest.mark.parametrize`` injects ``n_obs`` and ``n_samples`` from the
    list of tuples above — pytest calls this function once per tuple.
    """
    rng = np.random.default_rng(RNG_SEED)

    y_obs, y_pred, weights = _make_pit_inputs(
        y_obs_values=rng.standard_normal(n_obs),
        y_pred_values=rng.standard_normal((n_chain, n_draw, n_obs)),
        weights_values=_uniform_weights(n_obs, n_chain, n_draw),
    )
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

    ``@pytest.mark.parametrize`` injects ``true_mu`` and ``true_sigma`` from
    the list of tuples — pytest calls this function once per tuple.
    """
    rng = np.random.default_rng(RNG_SEED)

    y_obs, y_pred, weights = _make_pit_inputs(
        y_obs_values=rng.normal(true_mu, true_sigma, N_OBS),
        y_pred_values=rng.normal(true_mu, true_sigma, (N_CHAIN, N_DRAW, N_OBS)),
        weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
    )
    pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
    _, p = stats.kstest(pit, "uniform")
    assert p > 0.05, f"KS test failed for N({true_mu}, {true_sigma}): p={p:.4f}"


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
        idata, y_obs = _fit_oracle_normal()
        y_pred, weights = _extract_pred_and_weights(idata)
        pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
        return {"pit": pit, "weights": weights}

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
        exp_values = np.array([*np.arange(0.05, 0.96, 0.05).tolist(), 0.99, 1.0])
        expected_cov = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
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
        positives by chance. A well-calibrated model should rarely exceed 3 as
        binom.cdf(3, 21, 0.05) = 0.98.
        """

        rng = np.random.default_rng(RNG_SEED)
        pit = oracle_data["pit"]
        psis_weights = oracle_data["weights"]
        exp_values = np.array([*np.arange(0.05, 0.96, 0.05).tolist(), 0.99, 1.0])
        expected_cov = _convert_grid_numpy_to_dataarray(exp_values, exp_values)
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
    ) -> DataArray:
        """Fit model_fn on y_obs, return PIT values."""
        idata = model_fn(y_obs)
        var_name = _resolve_likelihood_var_name(idata)
        y_pred, weights = _extract_pred_and_weights(idata)
        return compute_loo_pit_model_agnostic(
            idata.observed_data[var_name], y_pred, weights
        )

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
        y_obs = stats.t.rvs(df=2, loc=0, scale=1, size=N_OBS, random_state=rng)

        def fit_normal_model(y_obs: np.ndarray):
            with pm.Model(coords={"obs": np.arange(len(y_obs))}):
                mu = pm.Normal("mu", mu=0.0, sigma=10.0)
                sigma = pm.HalfNormal("sigma", sigma=5.0)
                _ = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs, dims="obs")
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
        with pytest.warns(
            UserWarning, match="You should consider using a more robust model"
        ):
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
        y_obs = rng.normal(3.0, 1.0, size=N_OBS)

        def fit_biased_model(y_obs: np.ndarray):
            with pm.Model(coords={"obs": np.arange(len(y_obs))}):
                mu = pm.Normal("mu", mu=0.0, sigma=0.1)
                _ = pm.Normal(
                    "y",
                    mu=mu,
                    sigma=1.0,
                    observed=y_obs,
                    dims="obs",
                )
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
    """Smoke test: plot_loo_calibration_curve_with_reference runs end-to-end
    on a known-calibrated input and returns a valid CalibrationStats object.
    """
    rng = np.random.default_rng(RNG_SEED)

    mpl.use("Agg")

    monkeypatch.setattr(plt, "show", lambda: None)

    y_obs, y_pred, weights = _make_pit_inputs(
        y_obs_values=rng.standard_normal(N_OBS),
        y_pred_values=rng.standard_normal((N_CHAIN, N_DRAW, N_OBS)),
        weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
    )

    # Call the REAL function
    _, result = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs, y_pred=y_pred, weights=weights, n_boot=500, random_seed=RNG_SEED
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
    """Visual sanity check: plot the calibration curve for a known-calibrated
    input. The empirical coverage line should hug the 45-degree diagonal,
    with almost all points inside both uncertainty bands.

    Run with: pytest -m visual -s
    """
    rng = np.random.default_rng(RNG_SEED)

    mpl.use("Agg")

    y_obs, y_pred, weights = _make_pit_inputs(
        y_obs_values=rng.standard_normal(N_OBS),
        y_pred_values=rng.standard_normal((N_CHAIN, N_DRAW, N_OBS)),
        weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
    )

    fig, result = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs, y_pred=y_pred, weights=weights, random_seed=RNG_SEED
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

    MU: float = 0.0
    SIGMA: float = 1.0
    observed: Float64Matrix1D = np.random.default_rng(RNG_SEED).normal(
        loc=MU, scale=SIGMA, size=N_OBS
    )
    observations: ClassVar[dict[str, DataArray]] = {}
    predictions: ClassVar[dict[str, DataArray]] = {}
    weights: ClassVar[dict[str, DataArray]] = {}
    pits: ClassVar[dict[str, DataArray]] = {}
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

            y_obs, y_pred, w = _make_pit_inputs(
                y_obs_values=self.observed,
                y_pred_values=rng.normal(
                    loc=scen_mu,
                    scale=scen_sigma,
                    size=(N_CHAIN, N_DRAW, N_OBS),
                ),
                weights_values=_uniform_weights(N_OBS, N_CHAIN, N_DRAW),
            )

            self.observations[var_name] = y_obs
            self.predictions[var_name] = y_pred
            self.weights[var_name] = w
            self.pits[label] = compute_loo_pit_model_agnostic(y_obs, y_pred, w)

            _, result = plot_loo_calibration_curve_with_reference(
                y_obs=y_obs,
                y_pred=y_pred,
                weights=w,
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

        _ = plot_loo_calibration_curve_with_reference(
            y_obs=self.observations[var_name],
            y_pred=self.predictions[var_name],
            weights=self.weights[var_name],
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
