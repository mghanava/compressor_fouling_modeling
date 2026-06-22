"""
Unit test that verifies the LOO-based calibration curve sits on the diagonal
for a synthetic, perfectly-calibrated PyMC model.

The test works by:
  * generating data from a known Beta distribution,
  * fitting a *very weak* PyMC model (Uniform priors) so the posterior
    matches the data,
  * drawing posterior predictive samples,
  * reshaping them to the (n_obs, n_samples) format required by
    compute_loo_pit_model_agnostic,
  * passing **uniform** weights to that helper,
  * calling plot_loo_calibration_curve_with_reference and asserting that
    the calibration curve is (within Monte‑Carlo noise) on the 45° line.
"""

from pathlib import Path

import numpy as np
import pymc as pm

from compressor_fouling_modeling.utility import (
    plot_loo_calibration_curve_with_reference,
)

RNG_SEED = 42


# ----------------------------------------------------------------------
# Helper: build a ground‑truth, perfectly calibrated model
# ----------------------------------------------------------------------
def _build_ground_truth_model(seed: int = 0):
    """Return y_obs, posterior predictive draws and uniform weights.

    The model is intentionally misspecified with very weak priors, so the
    posterior recovers the true parameters → a well‑calibrated PIT.
    """
    rng = np.random.default_rng(seed)

    # ---- 1️⃣  True parameters (any positive values work) ----
    a_true, b_true = 2.5, 1.7
    n_obs = 1000
    y_obs = rng.beta(a_true, b_true, size=n_obs)

    # ---- 2️⃣  Simple PyMC model with weak priors ----
    with pm.Model() as model:
        a = pm.Uniform("a", lower=0, upper=10)
        b = pm.Uniform("b", lower=0, upper=10)
        _ = pm.Beta("y", alpha=a, beta=b, observed=y_obs)

        trace = pm.sample(
            draws=5000,
            tune=1000,
            cores=4,
            random_seed=seed,
            target_accept=0.90,
            progressbar=False,  # makes pytest less noisy
        )
        # ---- 3️⃣  Posterior predictive draws (n_samples, n_obs) ----
        ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=seed)
        trace.update(ppc)

    # print(f"ppc groups:{ppc}")
    # ---- 4️⃣  Prepare arguments for compute_loo_pit_model_agnostic ----
    pp = trace.posterior_predictive["y"].to_numpy()  # shape (n_samples, n_obs)
    n_obs = pp.shape[2]
    y_pred_flat = pp.reshape(-1, n_obs).T  # (n_obs, n_samples)
    # Uniform PSIS‑LOO weights – each draw gets the same weight.
    n_samples = y_pred_flat.shape[1]
    weights = np.full_like(y_pred_flat, 1.0 / n_samples)  # shape (n_obs, n_samples)

    return y_obs, y_pred_flat, weights


# ----------------------------------------------------------------------
# The actual test
# ----------------------------------------------------------------------
def test_loo_calibration_is_diagonal():
    """
    For a perfectly calibrated model the LOO PIT is Uniform[0, 1].
    Consequently the empirical coverage curve should be (near) identical to
    the expected coverage on the 45° line.
    """
    y_obs, y_pred, weights = _build_ground_truth_model(seed=RNG_SEED)

    # Call the *exact* function you wrote.  We keep the plot invisible in CI
    # by using a non‑interactive backend; the asserts will still run.
    fig, stats = plot_loo_calibration_curve_with_reference(
        y_obs=y_obs,
        y_pred=y_pred,
        weights=weights,
        # n_boot=10000,
        # ci_level=0.95,
        # figsize=(7, 7),
        random_seed=RNG_SEED,
    )
    # -----------------------------------------------------------------
    # Quick visual sanity check (optional, only when a display is available)
    # -----------------------------------------------------------------
    # If you run the test locally with a GUI, you can open the plot.
    file_path = Path(__file__).resolve()
    parent_dir = file_path.parent
    fname = Path(parent_dir / "calibration.png")
    _ = fig.savefig(fname)  # <-- uncomment for CI screenshot

    # --------------------------------------------------------------
    # Empirical vs. expected coverage – *dynamic* tolerance
    # --------------------------------------------------------------
    # Binomial (sampling) variance for a PIT empirical CDF at quantile q:
    #   var = q * (1 - q) / n_obs
    #   std = sqrt(var)
    # We take a 2-σ envelope (+10 % safety margin) and use the *maximum*
    # over all quantiles as the absolute tolerance.
    n_obs = y_obs.size
    exp_cov = stats.expected_coverage
    # variance of each empirical coverage estimate (approx.)
    var_q = exp_cov * (1.0 - exp_cov) / n_obs
    std_q = np.sqrt(var_q)
    # a conservative tolerance that works for *all* quantiles
    max_abs_err: float = 2.0 * std_q.max() * 1.10  # 2σ → 95 % CI, 10 % extra
    # If the sample size is tiny the envelope can be >1 – clip it.
    max_abs_err: float = min(max_abs_err, 0.5)

    print(f"max_abs_err {max_abs_err}")

    np.testing.assert_allclose(
        stats.empirical_coverage,
        exp_cov,
        atol=max_abs_err,
        rtol=0.0,
        err_msg=f"Empirical coverage deviates from the diagonal (tolerance={max_abs_err:.4f})",
    )

    # ------------------------------------------------------------------
    # Calibration errors should be tiny (Monte-Carlo noise only)
    # ------------------------------------------------------------------
    assert stats.calibration_error < std_q.min(), "Mean calibration error too large"
    assert stats.weighted_cal_error < std_q.max(), (
        "Weighted calibration error too large"
    )

    # ------------------------------------------------------------------
    # 3️⃣  No interval should be flagged as mis-calibrated
    # ------------------------------------------------------------------
    assert stats.n_miscalibrated == 0, (
        f"Found {stats.n_miscalibrated} miscalibrated points - "
        "the synthetic model should be perfectly calibrated"
    )


# def _build_ground_truth_model(n_obs: int, seed: int = 0):
#     """Return y_obs, posterior predictive draws and uniform weights.

#     The model is intentionally misspecified with very weak priors, so the
#     posterior recovers the true parameters → a well-calibrated PIT.
#     """
#     rng = np.random.default_rng(seed)

#     # ---- 1️⃣  True parameters (any positive values work) ----
#     a_true, b_true = 2.5, 1.7
#     y_obs = rng.beta(a_true, b_true, size=n_obs)

#     # ---- 2️⃣  Simple PyMC model with weak priors ----
#     with pm.Model() as model:
#         a = pm.Uniform("a", lower=0, upper=10)
#         b = pm.Uniform("b", lower=0, upper=10)
#         _ = pm.Beta("y", alpha=a, beta=b, observed=y_obs)

#         trace = pm.sample(
#             draws=5000,
#             tune=1000,
#             cores=4,
#             random_seed=seed,
#             target_accept=0.90,
#             progressbar=False,  # makes pytest less noisy
#         )
#         # ---- 3️⃣  Posterior predictive draws (n_samples, n_obs) ----
#         ppc = pm.sample_posterior_predictive(trace, var_names=["y"], random_seed=seed)
#         trace.update(ppc)
#         pm.compute_log_likelihood(trace)

#     # print(f"ppc groups:{ppc}")
#     # ---- 4️⃣  Prepare arguments for compute_loo_pit_model_agnostic ----
#     pp = trace.posterior_predictive["y"].to_numpy()  # shape (n_samples, n_obs)
#     n_obs = pp.shape[2]
#     y_pred_flat = pp.reshape(-1, n_obs).T  # (n_obs, n_samples)
#     # Uniform PSIS-LOO weights - each draw gets the same weight.
#     n_samples = y_pred_flat.shape[1]
#     weights = np.full_like(y_pred_flat, 1.0 / n_samples)  # shape (n_obs, n_samples)

#     return trace, y_obs, y_pred_flat, weights

# trace_gt, y_obs_gt, y_pred_gt, weights_gt = _build_ground_truth_model(n_obs=1000, seed=RANDOM_SEED)

# fig_good_calibration, stats_good_calibration = plot_loo_calibration_curves(trace_gt, RANDOM_SEED)
