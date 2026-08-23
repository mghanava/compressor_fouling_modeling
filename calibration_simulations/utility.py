"""Shared utilities for the calibration simulation study.

Collects generic, reusable helpers — timing, diagnostic plotting,
uniformity statistics, and system-sleep inhibition — that are not
specific to the spline model or the simulation pipeline itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import sys
import time
from types import TracebackType
from typing import TypedDict

from arviz_stats.loo import loo_pit
from arviz_stats.utils import ELPDData
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.stats import cramervonmises, differential_entropy, kstest
from scipy.stats._hypotests import CramerVonMisesResult
from scipy.stats._stats_py import KstestResult
from xarray import DataArray, Dataset, DataTree

# Local imports
from compressor_fouling_modeling.utility import (
    CalibrationStats,
    Float64Matrix1D,
    _resolve_likelihood_var_name,  # ruff: ignore[import-private-name] #pyright: ignore[reportPrivateUsage]
)


class LogTime:
    """A context manager for measuring execution time of code blocks and functions.

    Usage as a context manager:
    ```
    with ExecutionTimer(name="My Task"):
        # code to measure
        time.sleep(1)
    ```

    """

    def __init__(self, task_name: str):
        """Initialize the LogTime context manager with a task name."""
        self.start_time: float = time.time()
        self.end_time: float = 0.0
        self.execution_time: float = 0.0
        self.task_name: str = task_name

    def __enter__(self):
        """Start timing the execution of the code block.

        Returns:
            The context manager instance.

        """
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """End timing and print the execution time of the code block.

        Returns:
            False, so that any raised exception is propagated.

        """
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print(f"{self.task_name} executed in {self.execution_time:.3f} seconds.\n")
        return False  # Propagate any exceptions


class InhibitSleep:
    """Re-exec the process under systemd-inhibit to block system sleep.

    On first use, the process re-executes itself through
    `systemd-inhibit --what=sleep`, which keeps the system awake only
    while the inhibited process runs. Re-execution is skipped when the
    `_INHIBITED` environment variable is already set, preventing
    infinite recursion.

    Example:
        ```python
        with InhibitSleep("Running long simulations"):
            run_simulations()
        ```

    """

    ENV_MARKER: str = "_INHIBITED"

    def __init__(self, reason: str = "Running long-running task"):
        """Initialize the inhibit context with a sleep-inhibition reason.

        Args:
            reason: Human-readable reason passed to systemd-inhibit.
                Defaults to "Running long-running task".

        """
        self.reason: str = reason

    def __enter__(self) -> InhibitSleep:
        """Re-exec under systemd-inhibit unless already inhibited.

        Returns:
            The context manager instance.

        """
        if os.environ.get(self.ENV_MARKER):
            return self
        os.environ[self.ENV_MARKER] = "1"
        if shutil.which("systemd-inhibit"):
            os.execvp(
                "systemd-inhibit",
                [
                    "systemd-inhibit",
                    "--what=sleep",
                    f"--why={self.reason}",
                    sys.executable,
                    *sys.argv,
                ],
            )
        else:
            print(
                "Warning: systemd-inhibit not found, running without sleep inhibition.",
                file=sys.stderr,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Do nothing; sleep inhibition ends when the process exits.

        Returns:
            False so that any exception in the wrapped block is
            propagated.

        """
        return False


def plot_pit_hist(pit_data: Float64Matrix1D):
    """Plot a histogram of LOO-PIT values.

    Renders the distribution of leave-one-out probability integral
    transform values to visually assess uniformity, a hallmark of a
    well-calibrated predictive model.

    Args:
        pit_data: Array of LOO-PIT values in the unit interval.

    Returns:
        The matplotlib figure with the histogram drawn on it; the
        caller is responsible for saving or displaying it.

    Example:
        ```python
        fig = plot_pit_hist(calib.stats.loo_pit.to_numpy())
        fig.savefig("loo_pit_hist.png", dpi=150)
        ```

    """
    hist_fig, hist_ax = plt.subplots(figsize=(7, 4))
    _ = hist_ax.hist(pit_data, bins=30, edgecolor="black")
    _ = hist_ax.set_xlabel("LOO-PIT")
    _ = hist_ax.set_ylabel("Count")
    return hist_fig


def entropy_uniformity_z_score(
    rng: np.random.Generator,
    n: int,
    observed_entropy: float,
    n_simulations: int = 1000,
) -> tuple[float, float, float]:
    """Test observed differential entropy against a uniform null distribution.

    Simulates the differential entropy of perfectly uniform samples of
    size n to build a Monte Carlo null distribution, then locates the
    observed entropy within it via a z-score and a percentile rank. The
    result gauges whether the model's LOO-PIT entropy is consistent with
    uniformity, where lower entropy indicates less uniformity.

    Args:
        rng: Random number generator for the simulations.
        n: Number of observations per simulated uniform sample.
        observed_entropy: Differential entropy of the model's LOO-PIT
            values, in nats.
        n_simulations: Number of Monte Carlo replicates. Defaults to
            1000.

    Returns:
        A tuple (z_score, percentile, expected_entropy), where z_score
        is the observed entropy expressed in null-distribution standard
        deviations, percentile is the percentage of simulations with
        entropy at or below the observed value, and expected_entropy is
        the mean simulated entropy.

    Example:
        ```python
        z, pct, expected = entropy_uniformity_z_score(
            np.random.default_rng(42), n=500, observed_entropy=0.94
        )
        ```

    """
    simulated_entropies = np.array(
        [differential_entropy(rng.uniform(0, 1, n)) for _ in range(n_simulations)]
    )
    z_score = (
        observed_entropy - simulated_entropies.mean()
    ) / simulated_entropies.std()
    percentile = (simulated_entropies <= observed_entropy).mean() * 100
    return z_score, percentile, simulated_entropies.mean()


@dataclass
class FittedModel:
    """A fitted PyMC model and its associated artifacts."""

    model: pm.Model
    idata: DataTree
    fig: Figure | SubFigure


@dataclass
class LpvStats:
    """LOO-PIT latent probability variable data."""

    lpv_ds: Dataset
    lpv_da: DataArray


@dataclass
class UniformityStats:
    """Uniformity test results and entropy estimate for LOO-PIT values."""

    ks_result: KstestResult
    cvm_result: CramerVonMisesResult
    diff_entropy: float
    p_val_az: float
    p_val_coverage_az: float


@dataclass
class CalibrationResult:
    """Calibration-curve figure, stats, and derived p-values."""

    fig: Figure | SubFigure
    stats: CalibrationStats


def compute_loo_pit_uniformity_stats(
    loo_pit: Float64Matrix1D,
    idata: DataTree,
    rng: np.random.Generator,
    loo_diag: LooDiagnostics | None = None,
) -> tuple[UniformityStats, LpvStats]:
    """Run uniformity tests (KS, CvM) and estimate differential entropy on LOO-PIT.

    Args:
        loo_pit: Leave-one-out probability integral transform values.
        idata: InferenceData object containing the model trace and LOO diagnostics.
        rng: Random number generator for subsampling-based p-value estimation.
        loo_diag: LOO diagnostics object containing log weights and Pareto k values.

    Returns:
        A tuple of (UniformityStats, LpvStats) containing the uniformity test
        results and the LOO-PIT latent probability variable data.

    Example:
        ```python
        >>> uniformity_stats, lpv_stats = compute_loo_pit_uniformity_stats(loo_pit)
        ```

    """
    ks_result = kstest(loo_pit, "uniform")
    cvm_result = cramervonmises(loo_pit, cdf="uniform", args=(0, 1))
    diff_entropy = differential_entropy(loo_pit, method="correa")
    log_weights, pareto_k = (None, None)
    if loo_diag is not None:
        log_weights = loo_diag.elpd_data.log_weights.to_dataset()
        pareto_k = loo_diag.elpd_data.pareto_k.to_dataset()
    p_val_az, p_val_coverage_az, lpv_ds, lpv_da = calculate_pvalue_arviz(
        idata, rng, log_weights=log_weights, pareto_k=pareto_k
    )
    uniformity = UniformityStats(
        ks_result=ks_result,
        cvm_result=cvm_result,
        diff_entropy=diff_entropy,
        p_val_az=p_val_az,
        p_val_coverage_az=p_val_coverage_az,
    )
    lpv = LpvStats(lpv_ds=lpv_ds, lpv_da=lpv_da)
    return uniformity, lpv


def calculate_pvalue_arviz(
    idata: DataTree,
    rng: np.random.Generator,
    log_weights: Dataset | None = None,
    pareto_k: Dataset | None = None,
):
    """Compute the posterior predictive uniformity p-values from ArviZ.

    Uses LOO-PIT with Pareto smoothing to obtain calibrated posterior
    predictive quantiles, then applies a probability-of-cast uniformity
    test to both the raw LOO-PIT values and their coverage-transformed
    counterparts.

    Args:
        idata: Inference data tree containing posterior predictive draws.
        log_weights: LOO log weights per observation.
        pareto_k: Pareto shape parameter estimates.
        rng: Random generator for LOO-PIT sampling.

    Returns:
        A tuple (p_lpv, p_coverage, lpv_values), where p_lpv is the
        uniformity p-value of the raw LOO-PIT values, p_coverage is the
        uniformity p-value of the coverage-transformed values, and
        lpv_values is the raw LOO-PIT series for the likelihood variable. TO BE COMPLETED FOR DATASET ADDITION.

    Example:
        >>> idata = ...
        >>> p_lpv, p_coverage, lpv_da, lpv_ds = calculate_pvalue_arviz(
        ...     idata, log_weights, pareto_k, rng
        ... )

    """
    lpv = loo_pit(idata, log_weights=log_weights, pareto_k=pareto_k, random_state=rng)
    res = lpv.azstats.uniformity_test(method="pot_c", dim=list(lpv.dims))
    lpv_coverage = 2 * np.abs(lpv - 0.5)
    res_coverage = lpv_coverage.azstats.uniformity_test(
        method="pot_c", dim=list(lpv_coverage.dims)
    )
    var_name = _resolve_likelihood_var_name(idata)
    return (
        float(res[0][var_name]),
        float(res_coverage[0][var_name]),
        lpv,
        lpv[var_name],
    )


@dataclass
class ModelDims:
    """Dimensions identifying a specific model configuration."""

    m: int
    """Smoothing basis dimension."""
    k: int
    """Number of knots."""


@dataclass
class ModelFit:
    """A sampled PyMC model, its inference data, and parameter count."""

    model: pm.Model
    idata: DataTree
    n_params: int


@dataclass
class LooDiagnostics:
    """LOO-CV results and Pareto-k exceedance diagnostics."""

    elpd_data: ELPDData
    ypred_loo: DataArray
    loo_rsquared: Float64Matrix1D
    frac_k_above_good_k_psis: float
    frac_k_above_good_k_loo_exp: float


class SimulationResult(TypedDict):
    """Metrics computed by one run of the simulation."""

    sim: int
    model_params: int
    model_eff_params: float
    elpd: float
    loo_rsquared: float
    frac_k_above_good_k_psis: float
    frac_k_above_good_k_loo_exp: float
    n_miscalibrated: int
    calibration_error: float
    weighted_cal_error: float
    ks_pvalue: float
    ks_stat: float
    cvm_pvalue: float
    cvm_stat: float
    diff_entropy: float
    p_val_az: float
    p_val_coverage_az: float
