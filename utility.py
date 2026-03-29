"""Utility functions for compressor fouling modeling analysis.

This module provides tools for data preparation, model training, SHAP analysis,
residual analysis, CUSUM-based anomaly detection, and Bayesian regression evaluation.
"""

from dataclasses import dataclass
from enum import Enum
from itertools import groupby
from typing import Any, Literal, cast

import arviz_stats as azs
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import pytensor.tensor as pt
import scipy
import xarray as xr
from arviz_base import extract
from arviz_stats.base import array_stats
from jax import vmap
from jax.scipy.stats import norm as jax_norm
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LearningCurveDisplay, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xarray_einstats.stats import XrContinuousRV


def calculate_empirical_sigma_stats(
    X: pd.DataFrame, y: pd.Series, setpoints: list[int]
):
    """Calculate empirical standard deviations and set up informed priors"""
    empirical_stats = {}

    print("Empirical Standard Deviations by Setpoint:")
    print("=" * 60)

    for sp in setpoints:
        mask = np.isclose(X["Outlet_Pressure_SP"], sp, atol=1e-6)
        y_subset = y[mask]

        empirical_stats[sp] = {
            "std": y_subset.std(),
            "n": len(y_subset),
            "mean": y_subset.mean(),
        }

        print(
            f"Setpoint {sp:3d} psi: σ_unscaled = {y_subset.std():.3f} "
            f"(n={len(y_subset)})"
        )

    # Calculate overall statistics
    all_stds = [stats["std"] for stats in empirical_stats.values()]
    weights = [stats["n"] for stats in empirical_stats.values()]
    mean_std = np.average(a=all_stds, weights=weights)
    min_std = np.min(all_stds)
    max_std = np.max(all_stds)
    range_to_mean_ratio = (max_std - min_std) / mean_std

    print("\n" + "=" * 60)
    print(
        f"Overall: weighted mean σ_unscaled = {mean_std:.3f}, "
        f"range normalized by mean = {100 * range_to_mean_ratio:.2f}% "
        f"spread around mean = [{100 * (min_std - mean_std) / mean_std:.2f}%"
        f", {100 * (max_std - mean_std) / mean_std:.2f}%]"
    )

    return empirical_stats, mean_std, range_to_mean_ratio, min_std, max_std


def prepare_hierarchical_noise_args(X: pd.DataFrame):
    setpoint_unique, setpoint_index = np.unique(
        X["Outlet_Pressure_SP"], return_inverse=True
    )
    map_sp_to_idx = {sp: i for i, sp in enumerate(setpoint_unique)}
    return setpoint_unique, setpoint_index, map_sp_to_idx


def prepare_baseline_mask(
    data: pd.DataFrame,
    baseline_period: list[tuple[str, str]] | None = None,
    shutin_mask: pd.Series | None = None,
    n_init_samples: int | None = None,
    min_periods: int | None = None,
    rolling_mean_std_multiplier: float | None = None,
) -> pd.Series:
    baseline_mask = pd.Series(False, index=data.index)
    if baseline_period is None:
        data["outlet_pressue_sp_track"] = (
            data["Outlet_Pressure"] - data["Outlet_Pressure_SP"]
        ) / (data["Outlet_Pressure_SP"] + 1e-6)
        # use initial n_init_samples worth of data to establish baseline
        if n_init_samples is None:
            n_init_samples = 60
        if min_periods is None:
            min_periods = 3
        if rolling_mean_std_multiplier is None:
            rolling_mean_std_multiplier = 1.25
        baseline_mean = data["outlet_pressue_sp_track"].iloc[:n_init_samples].mean()
        baseline_std = data["outlet_pressue_sp_track"].iloc[:n_init_samples].std()
        if shutin_mask is not None:
            data["outlet_pressue_sp_track_rolled_mean"] = (
                data["outlet_pressue_sp_track"]
                .mask(shutin_mask)
                .rolling(window=min_periods, center=False)
                .mean()
                .bfill()
                .where(~shutin_mask)
            )
        else:
            data["outlet_pressue_sp_track_rolled_mean"] = (
                data["outlet_pressue_sp_track"]
                .rolling(window=min_periods, center=False)
                .mean()
                .bfill()
            )
        baseline_mask = (
            data["outlet_pressue_sp_track_rolled_mean"]
            > baseline_mean - rolling_mean_std_multiplier * baseline_std
        )
    if baseline_period is not None:
        for start, end in baseline_period:
            baseline_mask |= (data.index >= start) & (data.index < end)
    return baseline_mask


def calculate_data_masks(
    data: pd.DataFrame,
    baseline_period: list[tuple[str, str]] | None = None,
    n_init_samples: int | None = None,
    min_periods: int | None = None,
    rolling_mean_std_multiplier: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    shutin_mask = data["Outlet_Pressure_SP"] == 0
    baseline_mask = prepare_baseline_mask(
        data.copy(),
        baseline_period,
        shutin_mask,
        n_init_samples,
        min_periods,
        rolling_mean_std_multiplier,
    )
    return baseline_mask, shutin_mask


def prepare_model_input(
    data: pd.DataFrame,
    feature_names: list[str] | None,
    feature_engineering_allowed: bool = True,
    target_name: str = "Outlet_Pressure",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    data = data.dropna(subset=["Outlet_Pressure"])
    data = data.interpolate(method="time", limit=3).dropna()
    base_features = [col for col in data.columns if col != target_name]

    engineered_features = []
    if feature_engineering_allowed:
        # Re-introduce Temperature_Rise calculation
        data.loc[:, "Temperature_Rise"] = (
            data["Outlet_Temperature"] - data["Inlet_Temperature"]
        )
        # Capture non-linear physics relationships
        data.loc[:, "Inlet_Pressure_x_Flow"] = (
            data["Inlet_Pressure"] * data["Inlet_Flow_Rate"]
        )
        data.loc[:, "Temp_x_Flow"] = data["Inlet_Temperature"] * data["Inlet_Flow_Rate"]
        engineered_features = [
            "Temperature_Rise",
            "Inlet_Pressure_x_Flow",
            "Temp_x_Flow",
        ]

    if feature_names is None:
        # If no feature_names are provided, use base_features.
        # Add engineered_features only if feature_engineering_allowed is True.
        if feature_engineering_allowed:
            feature_names = base_features + engineered_features
        else:
            feature_names = base_features
    else:
        # If feature_names are provided, use them.
        # Add engineered_features only if feature_engineering_allowed is True.
        if feature_engineering_allowed:
            feature_names = feature_names + engineered_features

    # Remove duplicates, preserving order
    final_feature_names = []
    if feature_names is not None:
        for feature in feature_names:
            if feature not in final_feature_names:
                final_feature_names.append(feature)

    X = data[final_feature_names]
    y = data[target_name]

    return data, X, y


def visualize_correlations(
    ax, data: pd.DataFrame, feature_1: str, feature_2: str, aspect: str = "equal"
):
    ax.scatter(data[feature_1], data[feature_2])
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_aspect(aspect)


def visualize_imputation(ax, data: pd.DataFrame, feature: str):
    data[feature].hist(
        ax=ax, label="original", histtype="step", stacked=True, fill=False
    )
    data.interpolate(method="time", limit=3)[feature].hist(
        ax=ax, label="time", histtype="step", stacked=True, fill=False, linestyle="--"
    )
    data.interpolate(method="cubicspline", limit=3)[feature].hist(
        ax=ax,
        label="cubicspline",
        histtype="step",
        stacked=True,
        fill=False,
        linestyle=":",
    )
    data.fillna(data.median())[feature].hist(
        ax=ax, label="median", histtype="step", stacked=True, fill=False, linestyle="-."
    )
    ax.legend()
    ax.set_title(f"Imputation Methods -\n {feature}")


def visualize_data_folds(X, y, data_folds, fname: str):
    n_col = len(data_folds)
    fig, axes = pyplot.subplots(1, n_col, figsize=(20, 7))
    for i, (train_split_idx, test_split_idx) in enumerate(data_folds):
        X_train, X_test = X.iloc[train_split_idx], X.iloc[test_split_idx]
        y_train, y_test = y.iloc[train_split_idx], y.iloc[test_split_idx]

        train_sp_distribution = (
            X_train["Outlet_Pressure_SP"].value_counts().sort_index()
        )
        test_sp_distribution = X_test["Outlet_Pressure_SP"].value_counts().sort_index()

        # Get all unique indices from both distributions
        all_indices = sorted(
            set(train_sp_distribution.index) | set(test_sp_distribution.index)
        )

        # Reindex both distributions to have the same index
        train_sp_distribution = train_sp_distribution.reindex(all_indices, fill_value=0)
        test_sp_distribution = test_sp_distribution.reindex(all_indices, fill_value=0)

        # Calculate bar width and positions
        bar_width = 0.35
        x_pos = range(len(all_indices))

        axes[i].bar(
            x=[x - bar_width / 2 for x in x_pos],
            height=train_sp_distribution,
            width=bar_width,
            label=f"train {y_train.shape[0]} samples",
        )
        axes[i].bar(
            x=[x + bar_width / 2 for x in x_pos],
            height=test_sp_distribution,
            width=bar_width,
            label=f"test {y_test.shape[0]} samples",
        )

        # Set x-tick labels to show actual values
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(all_indices)

        axes[i].set_title(
            f"folder {i + 1}: target autocorrelations\n"
            f"train {y_train.autocorr():.3f} test: {y_test.autocorr():.3f}"
        )
        axes[i].legend()
    pyplot.savefig(fname, dpi=150, bbox_inches="tight")
    pyplot.close(fig)


def plot_timeseries_grid(data, config, fname, figsize=(30, 25)):
    fig, axs = pyplot.subplots(len(config), 1, sharex=True, figsize=figsize)

    for idx, series in enumerate(config):
        for col, color in series:
            axs[idx].plot(data.index, data[col], "o-", color=color, label=col)

        axs[idx].legend()

    pyplot.savefig(fname, dpi=150, bbox_inches="tight")
    pyplot.close(fig)


def visualize_learning_curve(
    estimator,
    X,
    y,
    train_sizes,
    cv,
    scoring,
    negate_score,
    shuffle,
    random_seed,
    fname: str,
):
    fig, ax = pyplot.subplots(1, 1)
    LearningCurveDisplay.from_estimator(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        negate_score=negate_score,
        shuffle=shuffle,
        random_state=random_seed,
        n_jobs=-1,
        ax=ax,
    )
    pyplot.savefig(fname, dpi=150, bbox_inches="tight")
    pyplot.close(fig)


def train_model(
    X_baseline: pd.DataFrame,
    y_baseline: pd.Series,
    cv,
    model_type: Literal["lr", "spline"] = "lr",
    n_trials: int = 100,
):
    def objecttive(trial):
        regressor_param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-3, 100),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
        if model_type == "lr":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", linear_model.ElasticNet(**regressor_param)),
                ]
            )
        elif model_type == "spline":
            spline_param = {
                "n_knots": trial.suggest_int("n_knots", 2, 4),
                "knots": trial.suggest_categorical("knots", ["uniform", "quantile"]),
                "degree": trial.suggest_int("degree", 1, 3),
                "include_bias": trial.suggest_categorical(
                    "include_bias", [True, False]
                ),
                "extrapolation": trial.suggest_categorical(
                    "extrapolation", ["constant", "linear", "continue"]
                ),
            }
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("spline", SplineTransformer(**spline_param)),
                    ("regressor", linear_model.ElasticNet(**regressor_param)),
                ]
            )
        else:
            raise ValueError("Invalid model_type")

        scores = cross_val_score(
            model,
            X_baseline,
            y_baseline,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        # Report intermediate value for pruning
        trial.report(scores.mean(), step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        return scores.mean()

    # create study and optimize
    study = optuna.create_study(
        direction="maximize",
        study_name="compressor_outlet_pressure",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(objecttive, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    regressor_param = {
        "alpha": float(trial.params["alpha"]),
        "l1_ratio": float(trial.params["l1_ratio"]),
        "fit_intercept": trial.params["fit_intercept"],
    }
    if model_type == "lr":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", linear_model.ElasticNet(**regressor_param)),
            ]
        )
    elif model_type == "spline":
        spline_param = {
            "n_knots": int(trial.params["n_knots"]),
            "knots": trial.params["knots"],
            "degree": int(trial.params["degree"]),
            "include_bias": trial.params["include_bias"],
            "extrapolation": trial.params["extrapolation"],
        }
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(**spline_param)),
                ("regressor", linear_model.ElasticNet(**regressor_param)),
            ]
        )
    else:
        raise ValueError("model_type must be 'lr' or 'spline'")
    pipeline.fit(X_baseline, y_baseline)
    # Save fitted model for later use
    best_model = pipeline
    # evaluate model on training data
    y_baseline_pred = best_model.predict(X_baseline)
    baseline_residuals = y_baseline.to_numpy() - y_baseline_pred
    mae = mean_absolute_error(y_baseline, y_baseline_pred)
    r2 = r2_score(y_baseline, y_baseline_pred)
    n = len(y_baseline)
    p = len(np.where(best_model.named_steps["regressor"].coef_ > 0)[0])
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"\nBaselibe Training MAE: {mae:.4f}")
    print(f"Baseline Training R^2: {r2:.4f}")
    print(f"Baseline Training adjusted R^2: {adjusted_r2:.4f}")
    print(f"Baseline Residuals Mean: {baseline_residuals.mean():.3f}")
    print(f"Baseline Residuals Std: {baseline_residuals.std():.3f}")
    return best_model


def calculate_residuals(model, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    y_pred = model.predict(X)
    residuals = y.to_numpy() - y_pred
    return pd.Series(data=residuals, index=y.index)


def find_anomaly_onsets(indices, min_consecutive=3) -> list:
    """Find the onset indices where at least min_consecutive consecutive indices follow.

    Parameters
    ----------
    indices: array-like, sorted list of indices
    min_consecutive: minimum number of consecutive indices required

    Returns
    -------
    list of onset indices that meet the criteria

    """
    onsets = []
    # Group consecutive numbers
    for _, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
        group = list(g)
        if len(group) >= min_consecutive:
            onsets.append(group[0][1])  # First element of the group

    return onsets


def calculate_snr(
    residuals: pd.Series,
    fouling_date: pd.Timestamp,
    baseline_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"],
) -> tuple[float, float]:
    """SNR= Magnitude of Fouling Drop (δ)/ Standard Deviation of Residuals (σ)."""
    baseline_residuals = residuals[baseline_mask]
    baseline_std = baseline_residuals.std()
    fouling_mask = (residuals.index >= fouling_date) & (~baseline_mask.to_numpy())
    fouling_residuals = residuals[fouling_mask]
    fouling_mean = (
        -fouling_residuals.mean()
        if anomaly_direction == "neg"
        else fouling_residuals.mean()
    )
    snr = fouling_mean / baseline_std
    fouling_max = (
        -fouling_residuals.min()
        if anomaly_direction == "neg"
        else fouling_residuals.max()
    )
    snr_peak = fouling_max / baseline_std
    return snr, snr_peak


def convert_index_to_date(
    fouling_idx: list[int], residuals: pd.Series
) -> list[pd.Timestamp]:
    fouling_dates = []
    if fouling_idx is not None:
        for idx in fouling_idx:
            fouling_date = residuals.index[idx]
            assert isinstance(fouling_date, pd.Timestamp), (
                "fouling date must have pandas timestamp type!"
            )
            fouling_dates.append(fouling_date)
    return fouling_dates


def calculate_memoryless_cusum(
    residuals: pd.Series,
    shutin_mask: pd.Series,
    drift: float,
    target_mean: float,
    anomaly_direction: Literal["pos", "neg"],
) -> np.ndarray:
    maintenance_resets = []
    cusum = np.zeros(len(residuals))
    for i in range(1, len(residuals)):
        current_date = residuals.index[i]
        # Reset CUSUM to 0 during the entire shut-in period
        if shutin_mask.iloc[i]:
            if not shutin_mask.iloc[i - 1]:  # Log start of maintenance
                maintenance_resets.append(current_date)
                print(f"CUSUM held at 0 during maintenance starting {current_date}")
            cusum[i] = 0
            continue

        # If it's the first day after a shut-in, reset to 0 before accumulating
        if not shutin_mask.iloc[i] and shutin_mask.iloc[i - 1]:
            cusum[i] = 0
        # Accumulate CUSUM during normal operation
        if anomaly_direction == "pos":
            cusum[i] = max(0, cusum[i - 1] + residuals.iloc[i] - target_mean - drift)
        else:
            cusum[i] = max(0, cusum[i - 1] - residuals.iloc[i] + target_mean - drift)
    return cusum


def plot_fouling_summary(
    y: pd.Series,
    y_pred: pd.Series,
    baseline_mask: pd.Series,
    shutin_mask: pd.Series,
    cusum: np.ndarray,
    alarm_mask: pd.Series,
    fname: str,
    threshold_multiplier: float = 3.0,
):
    """Plot a summary of the fouling detection analysis.

    This function generates a figure with three subplots:
    1. Actual vs. Baseline Prediction
    2. Residuals from Baseline Model
    3. CUSUM Control Chart

    Parameters
    ----------
    y : pd.Series
        Actual target values.
    y_pred : np.ndarray
        Predicted target values from the baseline model.
    baseline_mask : pd.Series
        Boolean series indicating baseline periods.
    threshold : float
        Decision threshold for residuals and CUSUM chart.
    shutin_mask : pd.Series
        Boolean series indicating shut-in periods.
    cusum_results : dict[str, np.ndarray]
        Dictionary containing 'cusum_high' and 'cusum_low' arrays.
    threshold_multiplier : float, optional
        Multiplier for standard deviation to set threshold, by default 3.0.

    """
    residuals = y - y_pred
    baseline_residuals = residuals[baseline_mask]
    threshold = threshold_multiplier * baseline_residuals.std()
    fig, axs = pyplot.subplots(3, 1, figsize=(25, 15), sharex=True)
    fig.suptitle("Anomaly Detection Analysis", fontsize=16, fontweight="bold")

    # 1. Actual vs Baseline Prediction
    axs[0].plot(y.index, y, linewidth=1, label="Actual")
    axs[0].plot(y.index, y_pred, linewidth=1, linestyle="--", label="Baseline Model")
    axs[0].set_title("Actual vs Baseline Prediction")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylabel("Outlet Pressure")

    ylim = axs[0].get_ylim()
    axs[0].fill_between(
        y.index,
        ylim[0],
        ylim[1],
        where=shutin_mask,
        facecolor="gray",
        alpha=0.5,
        label="Shut-in Period",
    )
    axs[0].set_ylim(ylim)
    axs[0].legend()

    # 2. Residuals from Baseline Model
    axs[1].scatter(residuals.index, residuals, s=2)
    axs[1].axhline(y=0, color="g", linestyle="-", linewidth=2, label="Target")
    axs[1].axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label=f"±{threshold_multiplier}σ Threshold",
    )
    axs[1].axhline(y=-threshold, color="r", linestyle="--")
    axs[1].grid(True, alpha=0.3)
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals from Baseline Model")
    axs[1].legend()

    # 3. CUSUM Control Chart
    axs[2].scatter(residuals.index, cusum, s=2)
    axs[2].axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label=f"Decision Threshold: : {threshold_multiplier} σ",
    )
    axs[2].set_ylabel("Cusum")
    axs[2].set_title("CUSUM Control Chart")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    ylim = axs[2].get_ylim()
    axs[2].fill_between(
        y.index,
        ylim[0],
        ylim[1],
        where=alarm_mask,
        facecolor="red",
        alpha=0.25,
        label="Alarm Period",
    )
    axs[2].set_ylim(ylim)
    axs[2].legend()

    pyplot.tight_layout()
    pyplot.savefig(fname, dpi=150, bbox_inches="tight")
    pyplot.close(fig)


def predict_fouling_onset(
    residuals: pd.Series,
    baseline_mask: pd.Series,
    shutin_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"],
    threshold_multiplier: float = 3,
    drift_multiplier: float = 0.5,
    target_mean: float = 0,
):
    # Baseline residuals
    # ~2.87 psi
    baseline_residuals = residuals[baseline_mask]
    # CUSUM calculation with resets during maintenance
    # We ignore any residual deviation smaller than drift (0.5 * 2.87 = 1.44 psi)
    drift = drift_multiplier * baseline_residuals.std()
    # If the cumulative sum of errors exceeds threshold (3 * 2.87 = 8.61 psi) value,
    # trigger the alarm
    threshold = threshold_multiplier * baseline_residuals.std()
    cusum = calculate_memoryless_cusum(
        residuals,
        shutin_mask,
        drift,
        target_mean,
        anomaly_direction,
    )
    # Calculate false alarm rate
    alarm_mask = pd.Series(cusum > threshold, index=residuals.index)
    false_alarm_rate = (baseline_mask & alarm_mask).sum() / sum(baseline_mask)
    print(f"\nFalse Alaram Rate for clean condition: {false_alarm_rate:.2%}")
    # Detect fouling onset
    fouling_idx = find_anomaly_onsets(np.where(alarm_mask)[0], min_consecutive=3)
    if fouling_idx is None:
        print("No fouling detected.")
        return None, cusum, alarm_mask
    else:
        fouling_dates = convert_index_to_date(fouling_idx, residuals)

        for i, date in enumerate(fouling_dates):
            print(f"Fouling number {i + 1} detected: {date}")
            # Calculate SNR
            snr, snr_peak = calculate_snr(
                residuals, date, baseline_mask, anomaly_direction
            )
            print(f"Signal-to-Noise Ratio: {snr:.2f}")
            print(f"Peak SNR: {snr_peak:.2f}")
        return fouling_dates, cusum, alarm_mask


@dataclass
class BayesianModelData:
    """Prepared and standardized data for Bayesian modeling."""

    X_scaled: pd.DataFrame
    X_mean: pd.Series
    X_std: pd.Series
    y_scaled: pd.Series
    y_mean: float
    y_std: float
    setpoint_timeseries: pd.Series
    sp_offset: pd.Series | None = None


def prepare_bayesian_model_args(
    X: pd.DataFrame,
    y: pd.Series,
    shuffle_baseline: bool = True,
    use_offset: bool = False,
    residual_target: bool = False,
    random_seed: int | None = None,
):
    """Prepare and standardize data for Bayesian modeling.

    Shuffles data to break autocorrelation and standardizes features/target
    based on the specified configuration.

    Args:
        X: Feature dataframe
        y: Target series
        use_offset: If True, compute setpoint offset
        residual_target: If True, use residual from setpoint as target
        random_seed: Random number generator

    Returns:
        BayesianModelData with scaled features, target, and metadata

    """
    X_shuffled = X
    y_shuffled = y
    # Shuffle to break autocorrelation
    if shuffle_baseline:
        X_shuffled, y_shuffled = _shuffle_data(X, y, random_seed)
    setpoint_timeseries = X_shuffled["Outlet_Pressure_SP"]

    # Scale features (excluding setpoint in offset/residual modes)
    X_scaled, X_mean, X_std = _scale_features(
        X_shuffled, exclude_setpoint=use_offset or residual_target
    )

    # Scale target based on mode
    if use_offset:
        y_scaled, y_mean, y_std = _scale_target(y_shuffled)
        sp_offset = (setpoint_timeseries - y_mean) / y_std
        _print_autocorr_change(y, y_scaled)
        return BayesianModelData(
            X_scaled=X_scaled,
            X_mean=X_mean,
            X_std=X_std,
            y_scaled=y_scaled,
            y_mean=y_mean,
            y_std=y_std,
            setpoint_timeseries=setpoint_timeseries,
            sp_offset=sp_offset,
        )

    elif residual_target:
        residual_obs = y_shuffled - setpoint_timeseries
        y_scaled, y_mean, y_std = _scale_target(residual_obs)

        _print_autocorr_change(y, y_scaled)
        return BayesianModelData(
            X_scaled=X_scaled,
            X_mean=X_mean,
            X_std=X_std,
            y_scaled=y_scaled,
            y_mean=y_mean,
            y_std=y_std,
            setpoint_timeseries=setpoint_timeseries,
        )

    else:
        y_scaled, y_mean, y_std = _scale_target(y_shuffled)
        _print_autocorr_change(y, y_scaled)
        return BayesianModelData(
            X_scaled=X_scaled,
            X_mean=X_mean,
            X_std=X_std,
            y_scaled=y_scaled,
            y_mean=y_mean,
            y_std=y_std,
            setpoint_timeseries=setpoint_timeseries,
        )


def _shuffle_data(
    X: pd.DataFrame, y: pd.Series, random_seed
) -> tuple[pd.DataFrame, pd.Series]:
    """Shuffle data to break autocorrelation."""
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]


def _scale_features(
    X: pd.DataFrame, exclude_setpoint: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize features (z-score normalization)."""
    if exclude_setpoint:
        X = X.loc[:, X.columns != "Outlet_Pressure_SP"]

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    return X_scaled, X_mean, X_std


def _scale_target(y: pd.Series) -> tuple[pd.Series, float, float]:
    """Standardize target variable (z-score normalization)."""
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std

    return y_scaled, y_mean, y_std


def _print_autocorr_change(y_original: pd.Series, y_scaled: pd.Series) -> None:
    """Print autocorrelation before and after shuffling/scaling."""
    print(
        f"autocorrelation changed from original {y_original.autocorr()} to "
        f"{y_scaled.autocorr()}."
    )


def calculate_cusum_with_uncertainty(
    residuals_distribution: np.ndarray,
    timestamps: pd.DatetimeIndex,
    shutin_mask: pd.Series,
    baseline_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"],
    drift_multiplier: float = 0.5,
    n_samples_subset: int | None = None,  # Use subset for speed
    random_seed: int | None = None,
) -> dict:
    n_samples_total, n_time = residuals_distribution.shape

    # Optionally subsample for speed
    if n_samples_subset is not None and n_samples_subset < n_samples_total:
        rng = (
            np.random.default_rng(random_seed)
            if random_seed is not None
            else np.random.default_rng()
        )
        sample_indices = rng.choice(n_samples_total, n_samples_subset, replace=False)
        residuals_subset = residuals_distribution[sample_indices]
        n_samples = n_samples_subset
        print(f"Using {n_samples} out of {n_samples_total} samples")
    else:
        residuals_subset = residuals_distribution
        n_samples = n_samples_total

    # Align masks
    shutin_aligned = shutin_mask.reindex(timestamps).fillna(False).to_numpy()
    baseline_aligned = baseline_mask.reindex(timestamps).fillna(False).to_numpy()

    # Per-sample parameters from baseline
    baseline_residuals = residuals_subset[:, baseline_aligned]
    target_means = baseline_residuals.mean(axis=1, keepdims=True)  # (n_samples, 1)
    drifts = drift_multiplier * baseline_residuals.std(
        axis=1, keepdims=True
    )  # (n_samples, 1)

    print(f"Target mean: {target_means.mean():.4f} ± {target_means.std():.4f}")
    print(f"Drift (k):   {drifts.mean():.4f} ± {drifts.std():.4f}")

    # Pre-compute increments for all samples
    if anomaly_direction == "pos":
        increments = residuals_subset - target_means - drifts
    else:
        increments = -residuals_subset + target_means - drifts

    # Find reset indices
    reset_after_shutin = np.zeros(n_time, dtype=bool)
    for i in range(1, n_time):
        if not shutin_aligned[i] and shutin_aligned[i - 1]:
            reset_after_shutin[i] = True

    # Calculate CUSUM iteratively (can't fully vectorize due to max(0, cumsum))
    cusum_paths = np.zeros((n_samples, n_time))

    for i in range(1, n_time):
        if shutin_aligned[i]:
            cusum_paths[:, i] = 0
        elif reset_after_shutin[i]:
            cusum_paths[:, i] = np.maximum(0, increments[:, i])
        else:
            cusum_paths[:, i] = np.maximum(0, cusum_paths[:, i - 1] + increments[:, i])

    # Summary statistics
    return {
        "cusum_paths": cusum_paths,
        "mean": cusum_paths.mean(axis=0),
        "median": np.median(cusum_paths, axis=0),
        "std": cusum_paths.std(axis=0),
        "ci_95": np.percentile(cusum_paths, [2.5, 97.5], axis=0),
        "ci_80": np.percentile(cusum_paths, [10, 90], axis=0),
        "ci_50": np.percentile(cusum_paths, [25, 75], axis=0),
        "timestamps": timestamps,
        "target_means": target_means.flatten(),
        "drifts": drifts.flatten(),
        "anomaly_direction": anomaly_direction,
    }


def compute_masked_statistics(
    residual_distribution: np.ndarray,
    timestamps: pd.DatetimeIndex,
    shutin_mask: pd.Series,
) -> dict:
    """Compute residual statistics, properly handling shut-in periods."""
    # Align mask
    shutin_aligned = shutin_mask.reindex(timestamps).fillna(False).to_numpy()
    operational_mask = ~shutin_aligned

    n_samples, n_time = residual_distribution.shape

    # Initialize with NaN for shut-in periods
    residual_mean = np.full(n_time, np.nan)
    residual_median = np.full(n_time, np.nan)
    residual_ci_95 = np.full((2, n_time), np.nan)
    residual_ci_80 = np.full((2, n_time), np.nan)
    residual_ci_50 = np.full((2, n_time), np.nan)

    # Compute statistics only for operational periods
    residual_mean[operational_mask] = residual_distribution[:, operational_mask].mean(
        axis=0
    )
    residual_median[operational_mask] = np.median(
        residual_distribution[:, operational_mask], axis=0
    )
    residual_ci_95[:, operational_mask] = np.percentile(
        residual_distribution[:, operational_mask], [2.5, 97.5], axis=0
    )
    residual_ci_80[:, operational_mask] = np.percentile(
        residual_distribution[:, operational_mask], [10, 90], axis=0
    )
    residual_ci_50[:, operational_mask] = np.percentile(
        residual_distribution[:, operational_mask], [25, 75], axis=0
    )

    return {
        "mean": residual_mean,
        "median": residual_median,
        "ci_95": residual_ci_95,
        "ci_80": residual_ci_80,
        "ci_50": residual_ci_50,
        "timestamps": timestamps,
        "operational_mask": operational_mask,
    }


def compute_exceedance_probability(cusum_paths, threshold):
    """Compute probability of exceeding threshold at each time point.

    Returns:
        p_above: P(CUSUM > +threshold)
        p_below: P(CUSUM < -threshold)
        p_either: P(|CUSUM| > threshold)

    """
    p_above = (cusum_paths > threshold).mean(axis=0)
    p_below = (cusum_paths < -threshold).mean(axis=0)
    p_either = (np.abs(cusum_paths) > threshold).mean(axis=0)

    return p_above, p_below, p_either


def visualize_probabilistic_cusum(
    residual_stats, cusum_result, timestamps, threshold, exceed_prob
):
    fig, axes = pyplot.subplots(4, 1, figsize=(16, 16), sharex=True)

    timestamps = timestamps
    op_mask = residual_stats["operational_mask"]

    # ===== Plot 1: CUSUM (already correct) =====
    ax = axes[0]
    ax.fill_between(
        timestamps,
        cusum_result["ci_95"][0],
        cusum_result["ci_95"][1],
        alpha=0.2,
        color="blue",
        label="95% CI",
    )
    # ax.fill_between(
    #     timestamps,
    #     cusum_result["ci_80"][0],
    #     cusum_result["ci_80"][1],
    #     alpha=0.3,
    #     color="blue",
    #     label="80% CI",
    # )
    # ax.fill_between(
    #     timestamps,
    #     cusum_result["ci_50"][0],
    #     cusum_result["ci_50"][1],
    #     alpha=0.5,
    #     color="blue",
    #     label="50% CI",
    # )
    ax.plot(timestamps, cusum_result["mean"], "b-", lw=2, label="Mean CUSUM")
    ax.axhline(
        threshold, color="red", linestyle="--", lw=2, label=f"h = {threshold:.2f}"
    )
    ax.axhline(0, color="black", linestyle="-", lw=1)
    ax.set_ylabel("CUSUM")
    ax.set_title("CUSUM with Posterior Uncertainty")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ===== Plot 2: Exceedance Probability =====
    ax = axes[1]
    ax.plot(timestamps, exceed_prob, "b-", lw=2, label="P(CUSUM > h)")
    ax.axhline(0.95, color="orange", linestyle="--", lw=2, label="95% threshold")
    ax.axhline(0.50, color="gray", linestyle=":", lw=1, label="50% threshold")
    # Shade shut-in periods
    for i in range(len(timestamps) - 1):
        if not op_mask[i]:
            ax.axvspan(timestamps[i], timestamps[i + 1], alpha=0.3, color="gray")
    ax.set_ylabel("Probability")
    ax.set_title("Probability of Exceeding Control Limits")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ===== Plot 3: Residuals (FILTERED) =====
    ax = axes[2]

    # Only plot where operational (NaN values will create gaps automatically)
    ax.fill_between(
        timestamps,
        residual_stats["ci_95"][0],
        residual_stats["ci_95"][1],
        alpha=0.3,
        color="blue",
        label="95% CI",
    )
    ax.plot(timestamps, residual_stats["mean"], "b-", lw=1, label="Mean residual")
    ax.axhline(0, color="black", linestyle="-", lw=1)

    # Shade shut-in periods
    for i in range(len(timestamps) - 1):
        if not op_mask[i]:
            ax.axvspan(
                timestamps[i],
                timestamps[i + 1],
                alpha=0.3,
                color="gray",
                label="Shut-in" if i == 0 else "",
            )

    ax.set_ylabel("Residual (scaled)")
    ax.set_title("Prediction Residuals (Operational Periods Only)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # ===== Plot 4: CUSUM Uncertainty Width =====
    ax = axes[3]
    cusum_width = cusum_result["ci_95"][1] - cusum_result["ci_95"][0]
    # Set shut-in periods to NaN for cleaner plot
    cusum_width_clean = cusum_width.copy()
    cusum_width_clean[~op_mask] = np.nan
    ax.plot(timestamps, cusum_width_clean, "purple", lw=2)
    ax.set_ylabel("95% CI Width")
    ax.set_xlabel("Time")
    ax.set_title("CUSUM Uncertainty (Operational Periods Only)")
    ax.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.savefig("cusum_filtered_visualization.png", dpi=150, bbox_inches="tight")
    pyplot.show()


class CoefPrior(Enum):
    """Enumeration of coefficient prior distributions for Bayesian regression.

    Attributes
    ----------
    NORMAL : str
        Normal (Gaussian) prior distribution.
    LAPLACE : str
        Laplace prior distribution for sparse solutions.

    """

    NORMAL = "normal"
    LAPLACE = "laplace"


class NoiseStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes
    ----------
    SINGLE : str
        Single global noise parameter across all observations.
    HIERARCHICAL : str
        Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class InterceptStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes
    ----------
    SINGLE : str
        Single global noise parameter across all observations.
    HIERARCHICAL : str
        Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class CoefficientStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes
    ----------
    SINGLE : str
        Single global noise parameter across all observations.
    HIERARCHICAL : str
        Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class LikeLiHood(Enum):
    NORMAL = "normal"
    T = "t-student"


def build_coords(
    X: pd.DataFrame,
    has_setpoint_coords: bool = True,
    hierarchical_kwargs: dict | None = None,
):
    coords = {
        "obs": np.arange(len(X)),
        "predictors": X.columns,
    }
    if has_setpoint_coords:
        assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
        # hierarchical_kwargs must contain "setpoint_unique"
        n_sp = len(hierarchical_kwargs["setpoint_unique"])
        coords["setpoint"] = np.arange(n_sp)
    return coords


def add_data(
    X: np.ndarray,
    y: np.ndarray,
    has_setpoint_coords: bool = True,
    hierarchical_kwargs: dict | None = None,
    offset=None,
):
    X_data = pm.Data("X", X, dims=("obs", "predictors"))
    y_data = pm.Data("y", y, dims="obs")
    offset_data = pm.Data("offset", offset, dims="obs") if offset is not None else None
    sp_idx_data = None
    if has_setpoint_coords:
        assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
        sp_idx_data = pm.Data(
            "sp_idx",
            hierarchical_kwargs["setpoint_index"],
            dims="obs",
        )

    return X_data, y_data, sp_idx_data, offset_data


def add_noise(
    noise_structure: NoiseStructure,
    noise_kwargs: dict[str, float],
    hierarchical_kwargs: dict[str, Any] | None = None,
    sp_idx_data=None,
    rng=None,
):
    """Add noise model to the Bayesian regression.

    Returns
    -------
    sigma_for_likelihood : tensor
        Sigma values to use in likelihood (indexed if hierarchical)
    sigma_by_setpoint : tensor or None
        The 4 unique sigma values (only for hierarchical)

    """
    # sigma (τ to avoid confusion) controls multiplicative uncertainty
    # log σ ∼ N(log(σ_{0}), τ) => σ ∼ LogNormal(log(σ_{0}), τ) =>
    # σ=σ_{0}⋅exp(ϵ), ϵ ∼ N(0,τ)
    # Suppose we believe "σ is likely within ±X% of σ₀" => τ = log(1 + X)
    if noise_structure == NoiseStructure.NON_HIERARCHICAL:
        # Aleatoric uncertainty (measurement noise)
        log_sigma = pm.Normal(
            "log_sigma",
            mu=np.log(noise_kwargs["sigma_mu_mu"]),
            sigma=noise_kwargs["sigma_mu_sd"],
            rng=rng,
        )
        sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
        return sigma  # Scalar, used directly in likelihood
    elif noise_structure == NoiseStructure.HIERARCHICAL:
        assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
        assert sp_idx_data is not None, "sp_idx_data required for hierarchical!"
        # group-level mean & sd
        log_sigma_mu = pm.Normal(
            "log_sigma_mu",
            mu=np.log(noise_kwargs["sigma_mu_mu"]),
            sigma=noise_kwargs["sigma_mu_sd"],
            rng=rng,
        )
        # Let the σ for each setpoint vary around the group mean, with expected
        # multiplicative differences of up double the σ spread (i.e., 12%), unless
        # the data indicate otherwise.
        log_sigma_sd = pm.HalfNormal(
            "log_sigma_sd", sigma=hierarchical_kwargs["sigma_sd_sd"], rng=rng
        )
        # non‐centered per‐setpoint raw
        log_sigma_raw = pm.Normal("log_sigma_raw", 0, 1, dims="setpoint", rng=rng)
        # per‐setpoint log‐σ
        log_sigma_sp = pm.Deterministic(
            "log_sigma_sp",
            log_sigma_mu + log_sigma_raw * log_sigma_sd,
            dims="setpoint",
        )
        # unique sigmas (stored in inference data)
        sigma = pm.Deterministic(
            "sigma",
            pt.exp(log_sigma_sp),
            dims="setpoint",
        )
        # Return indexed version for likelihood (not stored separately)
        return sigma[sp_idx_data]
    else:
        raise ValueError(f"Unknown noise_structure: {noise_structure}")


def add_intercept(intercept_sd, intercept_structure: InterceptStructure):
    if intercept_structure is InterceptStructure.HIERARCHICAL:
        mu_intercept = pm.Normal("mu_intercept", 0.0, intercept_sd)
        sigma_intercept = pm.HalfNormal("sigma_intercept", 0.5 * intercept_sd)
        return pm.Normal(
            "intercept", mu=mu_intercept, sigma=sigma_intercept, dims="setpoint"
        )
    else:
        return pm.Normal("intercept", mu=0.0, sigma=intercept_sd)


def add_coefficient_priors(
    coef_prior, coef_kwargs, coefficient_structure: CoefficientStructure
):
    """Add coefficient priors - shared or per-setpoint."""
    if coefficient_structure == CoefficientStructure.HIERARCHICAL:
        # Hierarchical per-setpoint betas
        # Typical effect size ~ within ±0.5
        beta_mu = pm.Normal(
            "beta_mu",
            mu=coef_kwargs.get("mu", 0.0),
            sigma=coef_kwargs.get("sd", 0.5),
            dims="predictors",
        )
        # Between-setpoint deviation ~ usually < 0.3
        beta_sd = pm.HalfNormal(
            "beta_sd", sigma=0.5 * coef_kwargs.get("sd", 0.5), dims="predictors"
        )
        # Non-centered parameterization
        beta_raw = pm.Normal("beta_raw", 0, 1, dims=("predictors", "setpoint"))
        beta = pm.Deterministic(
            "beta",
            beta_mu[:, None] + beta_raw * beta_sd[:, None],
            dims=("predictors", "setpoint"),
        )  # (n_predictors, n_setpoints)
        return beta, True

    else:
        # Shared betas
        if coef_prior is CoefPrior.NORMAL:
            return pm.Normal(
                "beta",
                mu=coef_kwargs.get("mu", 0.0),
                sigma=coef_kwargs.get("sd", 0.5),
                dims="predictors",
            ), False
        else:  # LAPLACE
            # Laplace variance = 2b²
            tau = pm.HalfNormal("tau", sigma=0.5 * np.sqrt(coef_kwargs.get("sd", 0.5)))
            return pm.Laplace(
                "beta",
                mu=coef_kwargs.get("mu", 0.0),
                b=tau,
                dims="predictors",
            ), False


def add_likelihood(
    intercept_structure,
    likelihood_model,
    intercept,
    beta,
    sigma,
    X_data,
    y_data,
    like_var: str = "y_like",
    sp_idx_data=None,
    likelihood_model_kwargs=None,
    offset_data=None,
    beta_has_setpoint_dim=False,
    rng=None,
):
    # Compute linear predictor
    if beta_has_setpoint_dim and sp_idx_data is not None:
        beta_obs = beta[:, sp_idx_data]  # (n_predictors, setpoint)
        mu_lin = (X_data * beta_obs.T).sum(axis=1)  # (n_predictors,)
    else:
        mu_lin = pt.dot(X_data, beta)  # (n_obs,)

    # Compute mean with intercept
    base = offset_data if offset_data is not None else 0.0

    if (
        intercept_structure == InterceptStructure.HIERARCHICAL
        and sp_idx_data is not None
    ):
        intercept_term = intercept[sp_idx_data]
    else:
        intercept_term = intercept

    mu = pm.Deterministic("mu", base + intercept_term + mu_lin, dims="obs")

    # Create likelihood
    if likelihood_model == LikeLiHood.NORMAL:
        return pm.Normal(
            like_var, mu=mu, sigma=sigma, observed=y_data, rng=rng, dims="obs"
        )
    elif likelihood_model == LikeLiHood.T:
        if likelihood_model_kwargs is None:
            raise ValueError(
                "likelihood_model_kwargs required for Student-T likelihood"
            )
        nu = pm.Gamma(
            "nu",
            alpha=likelihood_model_kwargs.get("alpha", 2.0),
            beta=likelihood_model_kwargs.get("beta", 0.1),
            rng=rng,
        )
        return pm.StudentT(
            like_var, nu=nu, mu=mu, sigma=sigma, observed=y_data, rng=rng, dims="obs"
        )
    else:
        raise ValueError(f"Unknown likelihood_model: {likelihood_model}")


def sample_and_build_idata(model, draws, tune, nuts_sampler, target_accept, rng):
    # Sample and assemble InferenceData
    with model:
        # draw the posterior
        idata = pm.sample(
            draws=draws,
            tune=tune,
            nuts_sampler=nuts_sampler,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=rng,
        )
        prior = pm.sample_prior_predictive(
            draws=draws, return_inferencedata=True, random_seed=rng
        )
        idata.extend(prior)
        # posterior predictive
        ppc = pm.sample_posterior_predictive(
            idata, return_inferencedata=True, random_seed=rng
        )
        idata.extend(ppc)
        pm.compute_log_likelihood(idata)
    return idata


def build_bayesian_model(
    X: pd.DataFrame,
    y: pd.Series,
    like_var: str = "y_like",
    has_setpoint_coords: bool = False,
    coef_prior: CoefPrior = CoefPrior.NORMAL,
    noise_structure: NoiseStructure = NoiseStructure.NON_HIERARCHICAL,
    intercept_structure: InterceptStructure = InterceptStructure.NON_HIERARCHICAL,
    coefficient_structure: CoefficientStructure = CoefficientStructure.NON_HIERARCHICAL,
    likelihood_model: LikeLiHood = LikeLiHood.NORMAL,
    intercept_sd: float = 1.0,
    coef_kwargs: dict[str, float] | None = None,
    noise_kwargs: dict[str, float] | None = None,
    hierarchical_kwargs: dict[str, Any] | None = None,
    likelihood_model_kwargs: dict[str, Any] | None = None,
    offset=None,
    random_seed: int | None = None,
):

    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    coef_kwargs = coef_kwargs or {}
    noise_kwargs = noise_kwargs or {}
    hierarchical_kwargs = hierarchical_kwargs or {}

    # Build coords once, up front
    coords = build_coords(X, has_setpoint_coords, hierarchical_kwargs)

    with pm.Model(coords=coords) as model:
        # Data containers
        X_data, y_data, sp_idx_data, offset_data = add_data(
            X.to_numpy(), y.to_numpy(), has_setpoint_coords, hierarchical_kwargs, offset
        )
        # Intercept and coefficient priors
        intercept = add_intercept(intercept_sd, intercept_structure)
        # print(f"coef_kwargs:{coef_kwargs}")
        beta, beta_has_setpoint_dim = add_coefficient_priors(
            coef_prior, coef_kwargs, coefficient_structure
        )
        # Noise model (single vs hierarchical)
        # print(f"noise_kwargs:{noise_kwargs}")
        sigma = add_noise(
            noise_structure, noise_kwargs, hierarchical_kwargs, sp_idx_data, rng
        )
        add_likelihood(
            intercept_structure,
            likelihood_model,
            intercept,
            beta,
            sigma,
            X_data,
            y_data,
            like_var,
            sp_idx_data,
            likelihood_model_kwargs,
            offset_data,
            beta_has_setpoint_dim,
            rng,
        )
    return model


def fit_bayesian_model(
    model,
    draws: int = 2000,
    tune: int = 1000,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "nutpie",
    target_accept: float = 0.9,
    random_seed: int | None = None,
):
    """Sample from an already-built model."""
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    idata = sample_and_build_idata(model, draws, tune, nuts_sampler, target_accept, rng)

    return idata


def compute_coverage(y_true, hdi_intervals):
    """Compute the proportion of true values that fall within their HDI intervals.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True observed values
    hdi_intervals : array-like, shape (n_samples, 2)
        HDI intervals where [:, 0] is lower bound and [:, 1] is upper bound

    Returns
    -------
    coverage : float
        Proportion of observations within their intervals (between 0 and 1)

    """
    lower = hdi_intervals[:, 0]
    upper = hdi_intervals[:, 1]

    within_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(within_interval)

    return coverage


def bayesian_bootstrap_sigma(
    y, random_seed: int | None = None, n_boot=100000
) -> np.ndarray:
    """Bayesian bootstrap to estimate the confidence interval of the standard deviation.

    The Bayesian bootstrap treats the observed data points as if they came from
    a discrete uniform distribution, then simulates the posterior distribution
    of any statistic (here, the standard deviation) by reweighting the data.
    Each bootstrap draw represents a different plausible empirical distribution
    consistent with the observed data. We are not resampling points but resampling
    distributions.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        The data points from which to estimate the standard deviation.
    n_boot : int, optional
        The number of bootstrap samples to generate. Default is 10,000.

    Returns
    -------
    ci : tuple
        A tuple containing the lower and upper bounds of the 95% confidence interval
        for the standard deviation.

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    n = len(y)
    # The weights must satisfy two critical constraints:
    # 1 - Non-negative: All weights w₁, w₂, ..., wₙ ≥ 0
    # 2 - Sum to 1: w₁ + w₂ + ... + wₙ = 1
    # These define what's called a probability simplex - the weights represent
    # a probability distribution over data points. Dirichlet distribution
    # automatically guarantees: w[i] ≥ 0 and sum(w) = 1
    # Generate all weights at once
    w = rng.dirichlet(np.ones(n), size=n_boot)  # shape (n_boot, n)
    # Calculates the mean using the random weights which is different from
    # classical bootstrap which samples with replacement
    mu = np.sum(w * y, axis=1)  # shape (n_boot,)
    var = np.sum(w * (y - mu[:, np.newaxis]) ** 2, axis=1)  # shape (n_boot, n)
    # Standard deviations
    return np.sqrt(var)


def bayesian_bootstrap_rmse(
    y_true, y_pred, random_seed: int | None = None, n_boot=100000
) -> np.ndarray:
    """Vectorized version for faster computation."""
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)

    residuals = y_true - y_pred
    squared_residuals = residuals**2

    # Generate all weights at once
    weights = rng.dirichlet(np.ones(n), size=n_boot)  # (n_boot, n)

    # Vectorized weighted MSE calculation
    # Each row: sum(weight * squared_residual)
    weighted_mse = np.sum(weights * squared_residuals, axis=1)  # (n_boot,)

    # RMSE for each bootstrap sample
    rmse_samples = np.sqrt(weighted_mse)

    return rmse_samples


def bayesian_bootstrap_band(
    loo_pit: np.ndarray,
    grid: np.ndarray,
    rng,
    ci_level: float = 0.95,
    B: int = 10000,
):
    """Compute Bayesian bootstrap uncertainty bands for a LOO calibration curve.

    This function estimates posterior uncertainty in the empirical calibration
    curve using the Bayesian bootstrap (Dirichlet reweighting).

    Parameters
    ----------
    loo_pit : array-like, shape (n,)
        LOO-PIT values for each observation. These should be computed from
        the leave-one-out predictive distribution:

            PIT_i = P(y_i^rep <= y_i | y_-i)

        Under perfect calibration, loo_pit ~ Uniform(0,1).
    grid : array-like, shape (m,)
        Grid of probability levels (e.g., np.linspace(0.05, 0.95, 19))
        at which the calibration curve is evaluated.
    rng : np.random.Generator
        Random number generator for reproducibility.
    ci_level : float, default=0.95
        Confidence level for the uncertainty bands (e.g., 0.95 for 95% bands).
    B : int, default=10000
        Number of Bayesian bootstrap draws (Dirichlet resamples).
        Larger values produce smoother and more stable uncertainty bands.

    Returns
    -------
    lower : ndarray, shape (m,)
        Lower bound of the 95% Bayesian bootstrap interval at each grid point.

    upper : ndarray, shape (m,)
        Upper bound of the 95% Bayesian bootstrap interval at each grid point.

    Notes
    -----
    The empirical calibration curve is:

        F_hat(q) = (1/n) * sum_i 1{loo_pit_i <= q}

    The Bayesian bootstrap replaces the uniform weights (1/n) with random
    Dirichlet(1,...,1) weights to propagate uncertainty in the empirical
    distribution of loo_pit.

    For each bootstrap draw b:

        F_b(q) = sum_i w_b,i * 1{loo_pit_i <= q}

    where w_b ~ Dirichlet(1,...,1).

    The returned bands correspond to the 2.5% and 97.5% quantiles of the
    bootstrap calibration curves across B draws.

    Interpretation
    --------------
    - The gray (binomial) band represents sampling variability under perfect
      calibration.
    - The Bayesian bootstrap band represents uncertainty in the estimated
      calibration curve given the observed PIT values.

    If the empirical calibration curve lies within the binomial band,
    there is no statistical evidence of miscalibration.

    This method is fully model-agnostic and does not assume any particular
    likelihood (Gaussian, Student-t, etc.).

    """
    loo_pit = np.asarray(loo_pit)
    n = loo_pit.size
    m = grid.size

    # Step 1: Indicator matrix (m × n)
    # I[j, i] = 1 if loo_pit[i] <= grid[j]
    I = (loo_pit[None, :] <= grid[:, None]).astype(float)

    # Step 2: Dirichlet weights (B × n)
    W = rng.dirichlet(np.ones(n), size=B)

    # Step 3: Compute all bootstrap curves at once where each element is a
    # weighted sample proportion
    # curves_{b,j} = sum_{i=1}^n w_b,i * 1(PIT_i <= q_j)
    # (B × n) @ (n × m) = (B × m)
    curves = W @ I.T

    # Step 4: Percentile bands
    alpha = 1 - ci_level
    lower = np.percentile(curves, 100 * alpha / 2, axis=0)
    upper = np.percentile(curves, 100 * (1 - alpha / 2), axis=0)

    return lower, upper


def _plot_model_vs_plugin(
    ax, setpoints, model_summary, plugin_sigma, plugin_ci, hdi_prob, raw_std
):
    """Primary calibration check: Does model σ match observed residual noise?"""
    # Model posterior
    means = [model_summary[sp]["mean"] for sp in setpoints]
    lower = [model_summary[sp]["hdi_lower"] for sp in setpoints]
    upper = [model_summary[sp]["hdi_upper"] for sp in setpoints]

    ax.fill_between(
        setpoints,
        lower,
        upper,
        alpha=0.3,
        color="C0",
        label=f"Model σ ({int(hdi_prob * 100)}% HDI)",
    )
    ax.plot(setpoints, means, "o-", color="C0", label="Model σ (posterior mean)")

    # Plug-in residual σ with CI
    plugin_vals = [plugin_sigma[sp] for sp in setpoints]
    for sp in setpoints:
        lo, hi = plugin_ci[sp]
        ax.vlines(sp, lo, hi, color="black", linewidth=2.5, alpha=0.7)
    ax.scatter(
        setpoints,
        plugin_vals,
        color="black",
        s=70,
        zorder=5,
        label="Residual σ at μ̂ (plug-in)",
    )
    ax.plot(
        setpoints,
        [raw_std[sp] for sp in setpoints],
        linestyle=":",
        color="gray",
        alpha=0.5,
        label="Raw σ (reference)",
    )
    ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title(
        "Primary Check: Model σ vs Observed Residuals\n"
        "(Black points should fall within blue band)",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def _plot_model_vs_perdraw(
    ax, setpoints, model_summary, perdraw_summary, hdi_prob, raw_std
):
    """Secondary check: Compare full posterior distributions."""
    # Model σ
    model_means = [model_summary[sp]["mean"] for sp in setpoints]
    model_lower = [model_summary[sp]["hdi_lower"] for sp in setpoints]
    model_upper = [model_summary[sp]["hdi_upper"] for sp in setpoints]

    ax.fill_between(
        setpoints,
        model_lower,
        model_upper,
        alpha=0.3,
        color="C0",
        label=f"Model σ ({int(hdi_prob * 100)}% HDI)",
    )
    ax.plot(setpoints, model_means, "o-", color="C0", label="Model σ (posterior mean)")

    # Per-draw residual σ
    res_means = [perdraw_summary[sp]["mean"] for sp in setpoints]
    res_lower = [perdraw_summary[sp]["hdi_lower"] for sp in setpoints]
    res_upper = [perdraw_summary[sp]["hdi_upper"] for sp in setpoints]

    ax.fill_between(
        setpoints,
        res_lower,
        res_upper,
        alpha=0.3,
        color="C1",
        label=f"Residual σ per draw ({int(hdi_prob * 100)}% HDI)",
    )
    ax.plot(setpoints, res_means, "s--", color="C1", label="Residual σ per draw")
    ax.plot(
        setpoints,
        [raw_std[sp] for sp in setpoints],
        linestyle=":",
        color="gray",
        alpha=0.5,
        label="Raw σ (reference)",
    )

    ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title(
        "Model σ vs Draw-wise Residual σ\n(Orange includes μ uncertainty → wider)",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def _plot_variance_decomposition(ax, setpoints, raw_std, plugin_sigma, model_summary):
    """Show how much variance the mean function explains."""
    raw_vals = [raw_std[sp] for sp in setpoints]
    residual_vals = [plugin_sigma[sp] for sp in setpoints]
    model_vals = [model_summary[sp]["mean"] for sp in setpoints]

    ax.plot(
        setpoints,
        raw_vals,
        "o--",
        color="gray",
        label="Raw SD (total variability)",
        linewidth=2,
    )
    ax.plot(
        setpoints,
        residual_vals,
        "o-",
        color="black",
        label="Residual SD (unexplained)",
        linewidth=2,
    )
    ax.plot(
        setpoints,
        model_vals,
        "o-",
        color="C0",
        label="Model σ (estimated noise)",
        linewidth=2,
    )

    # Shade the "explained" region
    ax.fill_between(
        setpoints,
        residual_vals,
        raw_vals,
        alpha=0.2,
        color="green",
        label="Variance explained by μ",
    )

    ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    ax.set_ylabel("Standard Deviation")
    ax.set_title(
        "Variance Decomposition\n(Gap = variance explained by mean function)",
        fontsize=11,
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)


def evaluate_noise_model(
    sigma_posterior,  # (chain, draw, [obs]) - posterior sigma samples
    mu_posterior,  # (chain, draw, obs) - posterior mean samples
    y_true,  # (obs,) - observed values (standardized)
    y_std,  # scalar - SD used for standardization
    y_mean,
    setpoint_timeseries,  # Series - setpoint per observation
    raw_std,  # dict - raw SD by setpoint
    random_seed,  # random number generator
    hdi_prob=0.95,
):
    """Evaluate noise model calibration by comparing three quantities.

    1. Model σ (posterior): The model's EXPLICIT belief about observation noise
    2. Residual σ (per draw): Implied noise from y - μ_draw (retains μ uncertainty)
    3. Residual σ (plug-in): std(y -μ) where μ = E[μ|data] as in frequentist regression

        Well-calibrated:   Model σ  ≈  Plug-in residual σ                     │
    │                         ↓              ↓                                │
    │                    (belief)    (observation)                            │
    │                                                                         │
    │  If plug-in residual σ >> model σ:                                      │
    │      → Model UNDERESTIMATES noise → overconfident predictions           │
    │                                                                         │
    │  If plug-in residual σ << model σ:                                      │
    │      → Model OVERESTIMATES noise → conservative predictions
    """
    setpoint_unique = np.unique(setpoint_timeseries)

    # =========================================================================
    # OVERALL MODEL PERFORMANCE (unstandardized scale)
    # =========================================================================
    mu_hat = mu_posterior.mean(axis=(0, 1))

    # Back-transform to original scale
    y_original = y_true * y_std + y_mean
    mu_hat_original = mu_hat * y_std + y_mean

    # Overall R²
    ss_tot = np.sum((y_original - y_original.mean()) ** 2)
    ss_res = np.sum((y_original - mu_hat_original) ** 2)
    r2_overall = 1 - ss_res / ss_tot

    # =========================================================================
    # VARIANCE DECOMPOSITION
    # =========================================================================
    # Total variance
    var_total = np.var(y_original)

    # Between-setpoint variance (variance of setpoint means)
    setpoint_means = {
        sp: y_original[setpoint_timeseries == sp].mean() for sp in setpoint_unique
    }
    grand_mean = y_original.mean()

    # Weighted between-group variance
    n_per_sp = {sp: (setpoint_timeseries == sp).sum() for sp in setpoint_unique}
    n_total = len(y_original)
    var_between = (
        sum(
            n_per_sp[sp] * (setpoint_means[sp] - grand_mean) ** 2
            for sp in setpoint_unique
        )
        / n_total
    )

    # Within-setpoint variance (pooled)
    var_within = (
        sum(
            n_per_sp[sp] * np.var(y_original[setpoint_timeseries == sp])
            for sp in setpoint_unique
        )
        / n_total
    )

    # =========================================================================
    # 1. MODEL'S POSTERIOR σ (the model's explicit noise estimate)
    # =========================================================================
    model_sigma_by_setpoint = {}

    # Handle different sigma structures (global vs per-obs)
    sigma_dims = len(sigma_posterior.dims)

    if sigma_dims == 2:  # (chain, draw) — global sigma
        # no dependency on setpoint; just added for the sake of scalability to
        # hierarchical noise model
        sigma_flat = sigma_posterior.to_numpy().flatten() * y_std
        for sp in setpoint_unique:
            model_sigma_by_setpoint[sp] = sigma_flat
    else:  # (chain, draw, obs) — per-observation sigma
        model_sigma_by_setpoint = dict(
            zip(
                setpoint_unique,
                sigma_posterior.mean(dim=("setpoint")).to_numpy() * y_std,
                strict=False,
            )
        )

    # =========================================================================
    # 2. Draw-wise residual (retains posterior uncertainty in μ)
    # =========================================================================
    # This shows what σ "would need to be" for each μ draw
    residuals_per_draw = y_true[None, None, :] - mu_posterior  # (chain, draw, obs)

    residual_sigma_per_draw = {}
    for sp in setpoint_unique:
        mask = (setpoint_timeseries == sp).to_numpy()
        sigma_draws = (
            residuals_per_draw[:, :, mask].std(axis=2).to_numpy().flatten() * y_std
        )
        residual_sigma_per_draw[sp] = sigma_draws
    # =========================================================================
    # 3. PLUG-IN RESIDUAL σ (point estimate at posterior mean μ̂)
    # =========================================================================
    # This is the classical "empirical" residual SD
    residuals_plugin = y_true - mu_hat

    plugin_sigma = {}
    plugin_sigma_ci = {}
    within_setpoint_r2 = {}

    for sp in setpoint_unique:
        mask = (setpoint_timeseries == sp).to_numpy()
        res_sp = residuals_plugin[mask]
        plugin_sigma[sp] = float(res_sp.std() * y_std)

        # Bootstrap CI for uncertainty on the plug-in estimate
        ci = azs.hdi(
            bayesian_bootstrap_sigma(np.asarray(res_sp), random_seed), prob=hdi_prob
        )
        plugin_sigma_ci[sp] = ci * y_std

        # Within-setpoint R²
        y_sp = y_original[mask]
        mu_sp = mu_hat_original[mask]
        ss_tot_sp = np.sum((y_sp - y_sp.mean()) ** 2)
        ss_res_sp = np.sum((y_sp - mu_sp) ** 2)
        within_setpoint_r2[sp] = 1 - ss_res_sp / ss_tot_sp if ss_tot_sp > 0 else 0.0

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    def summarize_draws(draws):
        hdi = azs.hdi(draws, prob=hdi_prob)
        return {
            "mean": float(np.mean(draws)),
            "median": float(np.median(draws)),
            "hdi_lower": float(hdi[0]),
            "hdi_upper": float(hdi[1]),
        }

    model_sigma_summary = {
        sp: summarize_draws(draws) for sp, draws in model_sigma_by_setpoint.items()
    }
    residual_sigma_per_draw_summary = {
        sp: summarize_draws(draws) for sp, draws in residual_sigma_per_draw.items()
    }

    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, axs = pyplot.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: THE KEY DIAGNOSTIC ---
    # Model σ vs Plug-in Residual σ
    _plot_model_vs_plugin(
        axs[0, 0],
        setpoint_unique,
        model_sigma_summary,
        plugin_sigma,
        plugin_sigma_ci,
        hdi_prob,
        raw_std,
    )

    # --- Panel 2: Model σ vs Per-Draw Residual σ ---
    # Shows effect of μ uncertainty
    _plot_model_vs_perdraw(
        axs[0, 1],
        setpoint_unique,
        model_sigma_summary,
        residual_sigma_per_draw_summary,
        hdi_prob,
        raw_std,
    )

    # --- Panel 3: Variance Decomposition ---
    _plot_variance_decomposition(
        axs[1, 0], setpoint_unique, raw_std, plugin_sigma, model_sigma_summary
    )

    # Panel 4: NEW - Overall variance decomposition pie/bar
    _plot_variance_breakdown(axs[1, 1], var_total, var_between, var_within, r2_overall)

    pyplot.tight_layout()

    # =========================================================================
    # PRINT DIAGNOSTICS
    # =========================================================================
    _print_diagnostics(
        setpoint_unique,
        model_sigma_summary,
        plugin_sigma,
        raw_std,
        within_setpoint_r2,
        r2_overall,
        var_between,
        var_within,
        var_total,
    )

    return {
        "model_sigma_summary": model_sigma_summary,
        "residual_sigma_per_draw_summary": residual_sigma_per_draw_summary,
        "plugin_sigma": plugin_sigma,
        "plugin_sigma_ci": plugin_sigma_ci,
        "r2_overall": r2_overall,
        "r2_within_setpoint": within_setpoint_r2,
        "variance_decomposition": {
            "total": var_total,
            "between_setpoint": var_between,
            "within_setpoint": var_within,
            "pct_between": var_between / var_total * 100,
            "pct_within": var_within / var_total * 100,
        },
        "fig": fig,
    }


def _plot_variance_breakdown(ax, var_total, var_between, var_within, r2_overall):
    """Visualize the variance decomposition."""
    # Bar chart showing decomposition
    categories = ["Total\nVariance", "Between\nSetpoint", "Within\nSetpoint"]
    values = [var_total, var_between, var_within]
    colors = ["steelblue", "seagreen", "coral"]

    ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.7)

    # Add percentage labels
    ax.text(
        1,
        var_between,
        f"{var_between / var_total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax.text(
        2,
        var_within,
        f"{var_within / var_total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    ax.set_ylabel("Variance")
    ax.set_title(
        f"Variance Decomposition\n(Overall R² = {r2_overall:.1%})", fontsize=11
    )
    ax.grid(alpha=0.3, axis="y")

    # Add annotation
    ax.annotate(
        f"Model captures {var_between / var_total * 100:.1f}% of variance\n"
        "(between-setpoint effect)",
        xy=(1, var_between),
        xytext=(1.5, var_total * 0.7),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
        ha="left",
    )


def _print_diagnostics(
    setpoints,
    model_summary,
    plugin_sigma,
    raw_std,
    within_r2,
    r2_overall,
    var_between,
    var_within,
    var_total,
):
    """Enhanced diagnostic output with variance decomposition context."""
    print("=" * 75)
    print("NOISE MODEL CALIBRATION DIAGNOSTICS")
    print("=" * 75)

    # Variance decomposition summary
    print("\n📊 VARIANCE DECOMPOSITION:")
    print("-" * 75)
    print(f"  Total variance:           {var_total:>10.2f}")
    print(
        f"  Between-setpoint:         {var_between:>10.2f}  "
        f"({var_between / var_total * 100:>5.1f}%)"
    )
    print(
        f"  Within-setpoint (pooled): {var_within:>10.2f}  "
        f"({var_within / var_total * 100:>5.1f}%)"
    )
    print(
        f"\n  Overall R²: {r2_overall:.3f} (model captures between-setpoint variation)"
    )

    # Per-setpoint table
    print("\n📋 PER-SETPOINT DIAGNOSTICS:")
    print("-" * 75)
    print(
        f"{'Setpoint':>10} | {'Model σ':>9} | {'Residual σ':>11} | "
        f"{'Ratio':>6} | {'Within R²':>9} | {'Raw SD':>8}"
    )
    print("-" * 75)

    ratios = []
    for sp in setpoints:
        model_s = model_summary[sp]["mean"]
        resid_s = plugin_sigma[sp]
        ratio = resid_s / model_s
        ratios.append(ratio)
        r2_sp = within_r2[sp]
        raw_s = raw_std[sp]

        # Flag interpretation
        flag = ""
        if ratio > 1.1:
            flag = " ⚠️"
        elif ratio < 0.9:
            flag = " 📉"

        print(
            f"{sp:>10.0f} | {model_s:>9.3f} | {resid_s:>11.3f} | "
            f"{ratio:>6.2f}{flag} | {r2_sp:>8.1%} | {raw_s:>8.3f}"
        )

    print("-" * 75)
    print(
        f"{'Mean':>10} | {model_summary[setpoints[0]]['mean']:>9.3f} | "
        f"{np.mean(list(plugin_sigma.values())):>11.3f} | {np.mean(ratios):>6.2f} |"
    )

    # Interpretation
    print("\n" + "=" * 75)
    print("📖 INTERPRETATION:")
    print("=" * 75)
    print(f"""
    Your model has R² = {r2_overall:.1%} overall, but low within-setpoint R².

    This is EXPECTED and CORRECT because:

    1. {var_between / var_total * 100:.1f}% of total variance is BETWEEN setpoints
       → Model mean function (μ) captures this well ✓

    2. {var_within / var_total * 100:.1f}% of total variance is WITHIN setpoints
       → This is mostly irreducible noise (sensor noise, minor fluctuations)
       → Model σ parameter captures this ✓

    The noise model evaluation above is checking whether model σ correctly
    estimates this within-setpoint noise — and with mean ratio = {np.mean(ratios):.2f},
    it does!

    ┌─────────────────────────────────────────────────────────────────────┐
    │  VERDICT: Well-specified model                                      │
    │  • Mean function captures systematic variation (between setpoints)  │
    │  • Sigma captures residual noise (within setpoints)                 │
    │  • Slight conservative bias (ratio < 1) is acceptable               │
    └─────────────────────────────────────────────────────────────────────┘
    """)


def plot_predictions_with_uncertainty(
    data, y_true, y_pred, hdi_prob: float = 0.95, dim: tuple = ("chain", "draw")
):
    # for each of the n_obs test data points, we have chain * draw posterior
    # predictive samples, and we want to compute one 95% HDI per test point
    hdi = azs.hdi(data, dim=dim, prob=hdi_prob)  # (n_obs,)
    coverage = compute_coverage(y_true, hdi.values)
    lower = hdi[:, 0]  # (n_obs,)
    upper = hdi[:, 1]  # (n_obs,)
    hdi_width = np.mean(
        upper - lower
    ).to_numpy()  # average uncertainty window around each prediction
    print("Average HDI width:", hdi_width)
    print(f"{int(hdi_prob * 100)}% HDI coverage:", coverage)
    pyplot.figure(figsize=(10, 5))
    pyplot.fill_between(
        np.arange(len(y_true)),
        lower,
        upper,
        alpha=0.3,
        label=f"{hdi_prob * 100}% HDI",
    )
    pyplot.plot(y_true, label="True", marker="o")
    pyplot.plot(y_pred, label="Mean Pred")
    pyplot.legend(loc="upper left")
    pyplot.title("Out-of-Sample Predictions with Uncertainty")
    pyplot.show()


@dataclass
class MetricsResult:
    """Container for computed metrics."""

    bayes_r2: np.ndarray
    residual_r2: np.ndarray
    loo_r2: np.ndarray
    mae_obs: np.ndarray
    rmse_obs: float
    rmse_obs_resampled: np.ndarray
    mae_posterior: np.ndarray | None = None
    mae_obs_az: str | None = None
    mae_loo: str | None = None
    rmse_posterior: np.ndarray | None = None
    rmse_obs_az: str | None = None
    rmse_loo: str | None = None


def _compute_metrics(
    idata,
    sample_dims,
    obs_dim,
    y_true,
    y_pred_samples,
    hdi_prob,
    like_var,
    mu_var,
    sigma_var,
    nu_var,
    likehood,
    rng,
) -> tuple[np.ndarray, pd.Series, np.ndarray, MetricsResult]:
    """Compute metrics."""
    dim = next(
        d for d in y_pred_samples.dims if d not in obs_dim
    )  # the stacked dimension name used for ("chain", "draw")
    y_pred = y_pred_samples.mean(dim=dim).to_numpy()
    residuals_obs = y_true - y_pred
    residuals_draw = y_true - y_pred_samples.transpose(dim, obs_dim).to_numpy()
    idata = calculate_modeled_residual_var(
        idata, mu_var, sigma_var, nu_var, likehood, obs_dim
    )
    metrics = MetricsResult(
        # all variance terms come from the model, and not directly from the data
        bayes_r2=azs.bayesian_r2(  # type: ignore
            idata,
            pred_mean=mu_var,
            group="posterior",
            summary=True,
            scale="var_residual",
            ci_kind="hdi",
            ci_prob=hdi_prob,
            round_to="4g",
        ),
        # computes residual variance from the observed data
        residual_r2=azs.residual_r2(  # type: ignore
            idata,
            pred_mean=mu_var,
            group="posterior",
            summary=True,
            ci_kind="hdi",
            ci_prob=hdi_prob,
            round_to="4g",
        ),
        mae_posterior=np.abs(residuals_draw).mean(axis=1),
        mae_obs=np.abs(residuals_obs),
        rmse_posterior=np.sqrt((residuals_draw**2).mean(axis=1)),
        rmse_obs=np.sqrt((residuals_obs**2).mean()),
        rmse_obs_resampled=bayesian_bootstrap_rmse(y_true, y_pred, rng),
        loo_r2=azs.loo_r2(  # type: ignore
            idata,
            var_name=like_var,
            summary=True,
            round_to="4g",
            ci_kind="hdi",
            ci_prob=0.95,
        ),
        mae_obs_az=azs.metrics(  # type: ignore
            idata, kind="mae", sample_dims=sample_dims, round_to="3g"
        ),
        mae_loo=azs.loo_metrics(idata, kind="mae", round_to="3g"),  # type: ignore
        rmse_obs_az=azs.metrics(  # type: ignore
            idata, kind="rmse", sample_dims=sample_dims, round_to="3g"
        ),
        rmse_loo=azs.loo_metrics(idata, kind="rmse", round_to="3g"),  # type: ignore
    )

    return y_pred, residuals_obs, residuals_draw, metrics


def _print_metrics(metrics: MetricsResult, hdi_prob: float) -> None:
    """Print performance metrics."""
    print("Model performance:")
    print(f"Bayesian R²: {metrics.bayes_r2}")
    print(f"Residual R²: {metrics.residual_r2}")
    print(
        f"Bayesian posterior MAE: mean={metrics.mae_posterior.mean():.4f}, "
        f"95% HDI={azs.hdi(metrics.mae_posterior, prob=hdi_prob)}"
    )
    print(
        f"Bayesian pointwise MAE: mean={metrics.mae_obs.mean():.4f}, "
        f"95% HDI={azs.hdi(metrics.mae_obs, prob=hdi_prob)}"
    )
    print(f"Arviz built-in MAE: {metrics.mae_obs_az}")
    print(
        f"Bayesian posterior RMSE: mean={metrics.rmse_posterior.mean():.4f}, "
        f"95% HDI={azs.hdi(metrics.rmse_posterior, prob=hdi_prob)}"
    )
    print(
        f"Bayesian pointwise RMSE: mean={metrics.rmse_obs:.4f}, "
        f"95% HDI={azs.hdi(metrics.rmse_obs_resampled, prob=hdi_prob)}"
    )
    print(f"Arviz built-in RMSE: {metrics.rmse_obs_az}")
    print(f"Leave-one-out Cross-validation adjusted R^2: {metrics.loo_r2}")
    print(f"Leave-one-out Cross-validation MAE: {metrics.mae_loo}")
    print(f"Leave-one-out Cross-validation RMSE: {metrics.rmse_loo}")


def _plot_mae_distributions(
    metrics: MetricsResult, hdi_prob: float, figsize: tuple
) -> None:
    """Plot MAE distribution histograms."""
    _, axs = pyplot.subplots(1, 2, figsize=figsize)

    axs[0].hist(metrics.mae_posterior)
    axs[0].set_xlabel("Mean Absolute Error per Draw")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title(
        "Distribution of MAE across Posterior Draws\nPosterior Uncertainty\n"
        f"mean={metrics.mae_posterior.mean():.4f}, "
        f"95% HDI={azs.hdi(metrics.mae_posterior, prob=hdi_prob)}",
        fontsize=10,
    )

    axs[1].hist(metrics.mae_obs)
    axs[1].set_xlabel("Mean Absolute Error per Observation")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title(
        "Distribution of MAE across Observations\nObservation-Level Prediction "
        f"Uncertainty\nmean={metrics.mae_obs.mean():.4f}, "
        f"95% HDI={azs.hdi(metrics.mae_obs, prob=hdi_prob)}",
        fontsize=10,
    )
    pyplot.show()


def _plot_residuals(
    residuals_obs: pd.Series,
    fitted_values: np.ndarray,
    hdi_width: float,
    setpoint_timeseries: pd.Series,
    figsize: tuple,
) -> None:
    """Plot residuals vs fitted values with HDI bounds.

    This diagnostic plot helps identify:
    - Non-random patterns (suggests model misspecification)
    - Heteroscedasticity (funnel shape)
    - Outliers
    - Whether residuals are similar across setpoints

    Args:
        residuals_obs: Observed residuals
        fitted_values: Model fitted/predicted values
        hdi_width: Width of highest density interval
        setpoint_timeseries: Setpoint values for each observation
        dataset_type: Type of dataset (e.g., "Train", "Test")

    """
    # Create figure
    _, ax = pyplot.subplots(figsize=figsize)

    # Get unique setpoints for coloring
    setpoints = np.unique(setpoint_timeseries)

    # Plot residuals colored by setpoint
    for sp in setpoints:
        mask = np.isclose(setpoint_timeseries, sp, atol=1e-6)
        ax.scatter(
            fitted_values[mask],
            residuals_obs[mask],
            label=f"SP = {int(sp)}",
            alpha=0.6,
            s=50,
        )

    # Add reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0, label="Zero", zorder=1)
    ax.axhline(
        y=hdi_width / 2,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="HDI half-width",
        zorder=1,
    )
    ax.axhline(y=-hdi_width / 2, color="red", linestyle="--", linewidth=1.5, zorder=1)

    # Labels and title
    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residual", fontsize=12)
    ax.set_title("Residuals vs Fitted Values", fontsize=14, fontweight="bold")

    # Legend with your preferred settings
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        framealpha=0.95,
        facecolor="white",
        edgecolor="gray",
    )

    # Add grid for easier pattern detection
    ax.grid(True, alpha=0.75, linestyle=":", linewidth=0.5)

    pyplot.tight_layout()


def _plot_autocorrelation(draw_autocorrs, time_series_index, data) -> None:
    _, axs = pyplot.subplots(1, 2, figsize=(15, 7))
    axs[0].hist(draw_autocorrs)
    axs[0].set_title(
        "Distribution of Lag-1 Autocorrelation Plot across all the posterior draws",
        fontsize=12,
        fontweight="bold",
    )
    pd.plotting.autocorrelation_plot(
        pd.Series(index=time_series_index, data=data), ax=axs[1]
    )
    axs[1].set_title(
        "Autocorrelation Plot for the posterior draw with maximum Lag-1 "
        "autocorrelation",
        fontsize=12,
        fontweight="bold",
    )
    pyplot.tight_layout()


def _compute_hdi_width(
    hdi_data, sample_dims: str | tuple[str, str], hdi_prob: float
) -> float:
    """Compute average HDI width from posterior predictive data."""
    hdi = azs.hdi(hdi_data, dim=sample_dims, prob=hdi_prob)
    lower, upper = hdi[:, 0], hdi[:, 1]
    return np.mean(upper - lower).to_numpy()


def evaluate_model_performance(
    idata,
    y_true: np.ndarray,
    y_pred_samples,
    setpoint_timeseries: pd.Series,
    sample_dims: tuple = ("chain", "draw"),
    obs_dim: str = "obs",
    like_var: str = "y_like",
    mu_var: str = "mu",
    sigma_var: str = "sigma",
    nu_var: str | None = None,
    likehood: LikeLiHood = LikeLiHood.NORMAL,
    hdi_prob: float = 0.95,
    random_seed: int | None = None,
    figsize: tuple = (10, 5),
) -> None:
    """Evaluate and visualize model performance for training or test datasets.

    Parameters
    ----------
    idata : InferenceData
        ArviZ InferenceData object containing posterior samples.
    dim : str
        Dimension name for sampling.
    mu : array-like
        Posterior mean predictions.
    y_true : array-like
        True target values.
    y_pred_samples : array-like
        Posterior predictive or Prediction (OOS) samples.
    dataset_type : string
        Type of dataset
    hdi_prob : float, default=0.95
        Probability mass for HDI computation.
    rng : Generator, optional
        Random number generator for bootstrap resampling.

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    y_pred, residuals_obs, residuals_draw, metrics = _compute_metrics(
        idata,
        sample_dims,
        obs_dim,
        y_true,
        y_pred_samples,
        hdi_prob,
        like_var,
        mu_var,
        sigma_var,
        nu_var,
        likehood,
        rng,
    )
    draw_autocorrs = np.abs(compute_autocorr(residuals_draw))

    _print_metrics(metrics, hdi_prob)
    _plot_mae_distributions(metrics, hdi_prob, figsize)
    hdi_data = idata.posterior_predictive[like_var]
    hdi_width = _compute_hdi_width(hdi_data, sample_dims, hdi_prob)
    _plot_residuals(residuals_obs, y_pred, hdi_width, setpoint_timeseries, figsize)

    _plot_autocorrelation(
        draw_autocorrs,
        setpoint_timeseries.index,
        residuals_draw[np.argmax(draw_autocorrs)],
    )


@dataclass
class MCMCSamples:
    """Container for MCMC samples in different formats.

    Attributes:
        mu_raw: Mean samples with dimensions (chain, draw, obs)
        mu_stacked: Mean samples with dimensions (sample, obs)
        sigma_raw: Noise parameter with dimensions (chain, draw), None for test data
        y_obs: Observed values
        y_pred_raw: Predicted values with dimensions (chain, draw, obs)
        y_pred_stacked: Predicted values with dimensions (sample, obs)
        log_likelihood_stacked: Log Likelihood with dimensions (n_obs, n_samples)

    """

    mu_raw: xr.DataArray  # (chain, draw, obs)
    mu_stacked: xr.DataArray  # (obs, n_samples)
    sigma_raw: xr.DataArray  # (chain, draw)
    sigma_stacked: xr.DataArray  # (n_samples,)
    y_pred_raw: xr.DataArray  # (chain, draw, obs)
    y_pred_stacked: xr.DataArray  # (obs, n_samples)
    y_obs: np.ndarray  # (obs, )
    log_likelihood_stacked: np.ndarray  # (n_obs, n_samples)


def _stack_samples(da):
    """Stack (chain, draw) dims that actually exist in the DataArray."""
    dims_to_stack = [d for d in ("chain", "draw") if d in da.dims]
    if not dims_to_stack:
        raise ValueError(f"No chain/draw dims found. Got: {da.dims}")
    return da.stack(sample=tuple(dims_to_stack))  # noqa: PD013 - stack() required for xarray dimension stacking


def exctract_pymc_groups_data(idata, obs_dim: str = "obs") -> MCMCSamples:
    # Extract mu and sigma
    mcmc_samples = extract(
        idata,
        group="posterior",
        var_names=["sigma", "mu"],
        combined=False,
    )
    mu_raw = mcmc_samples.mu
    mu_raw = cast(xr.DataArray, mu_raw)
    mu_stacked = _stack_samples(mcmc_samples.mu)
    sigma_raw = mcmc_samples.sigma
    sigma_stacked = _stack_samples(mcmc_samples.sigma)
    # Exctract observed values
    y_obs = extract(idata, group="observed_data", sample_dims=obs_dim).to_numpy()
    # Extract y predictions
    y_pred_raw = extract(idata, group="posterior_predictive", combined=False)
    y_pred_raw = cast(xr.DataArray, y_pred_raw)
    y_pred_stacked = _stack_samples(y_pred_raw)
    # Exctract log likelihood
    log_lik_flat = extract(idata, group="log_likelihood", combined=True).to_numpy()
    return MCMCSamples(
        mu_raw=mu_raw,
        mu_stacked=mu_stacked,
        sigma_raw=sigma_raw,
        sigma_stacked=sigma_stacked,
        y_pred_raw=y_pred_raw,
        y_pred_stacked=y_pred_stacked,
        y_obs=y_obs,
        log_likelihood_stacked=log_lik_flat,
    )


def check_likelihood_qqplot(
    idata,
    likelihood: Literal["normal", "t-student"],
    y_true: np.ndarray,
    dim: tuple = ("chain", "draw"),
):
    var_names = ["sigma", "mu", "nu"] if likelihood == "t-student" else ["sigma", "mu"]
    mcmc_samples = extract(
        idata,
        group="posterior",
        var_names=var_names,
        sample_dims=dim,
        combined=False,
    )
    sigma_mean = mcmc_samples.sigma.mean(dim=dim)  # scaler
    mu_mean = mcmc_samples.mu.mean(dim=dim)  # (n_obs, )
    sparams = None
    if likelihood == "t-student":
        nu_mean = mcmc_samples.nu.mean(dim=dim)  # scaler
        sparams = (nu_mean,)
    # standardize residuals (should be ~Normal(0, 1))
    residuals = (y_true - mu_mean) / sigma_mean
    fig, ax = pyplot.subplots(1, 1)
    scipy.stats.probplot(
        residuals.to_numpy(),
        dist="norm",
        sparams=sparams,
        plot=ax,
        fit=True,
        rvalue=True,
    )
    ax.set_title(f"Q-Q Plot of Residuals Given {likelihood} likelihood", fontsize=12)
    return fig


def select_top_bayesian_model(
    model_configs: list[dict[str, Any]], common_params: dict[str, Any]
) -> tuple:

    # Build and fit all models
    models = {}
    idatas = {}

    for config in model_configs:
        name = config["name"]

        # Build model
        model = build_bayesian_model(
            X=config["X"],
            y=config["y"],
            like_var=config.get("like_var", "y_like"),
            has_setpoint_coords=config.get("has_setpoint_coords", False),
            coef_prior=config.get("coef_prior", CoefPrior.NORMAL),
            noise_structure=config.get(
                "noise_structure", NoiseStructure.NON_HIERARCHICAL
            ),
            intercept_structure=config.get(
                "intercept_structure", InterceptStructure.NON_HIERARCHICAL
            ),
            coefficient_structure=config.get(
                "coefficient_structure", CoefficientStructure.NON_HIERARCHICAL
            ),
            likelihood_model=config.get("likelihood_model", LikeLiHood.NORMAL),
            offset=config.get("offset", None),
            **common_params,
        )
        print(f"Model {name} built successfully!")
        # fit model
        idata = fit_bayesian_model(
            model, nuts_sampler="nutpie", random_seed=common_params["random_seed"]
        )
        print(f"Model {name} fitted successfully!\n")

        models[name] = model
        idatas[name] = idata

    # Compare models
    print("================= Comparison of models =================")
    compare_results = azs.compare(idatas, method="BB-pseudo-BMA", round_to=2)
    display(compare_results)

    # Select best model and its idata
    best_model_name = compare_results.index[0]  # Top-ranked model
    model_best = models[best_model_name]
    idata_best = idatas[best_model_name]

    return model_best, idata_best, models, idatas


def calculate_modeled_residual_var(
    idata,
    mu_var: str = "mu",
    sigma_var: str = "sigma",
    nu_var: str | None = None,
    likehood: LikeLiHood = LikeLiHood.NORMAL,
    obs_dim="obs",
):
    # Get mu and sigma from posterior
    mu = idata.posterior[mu_var]  # shape: (chain, draw, obs)
    sigma = idata.posterior[sigma_var]  # shape: (chain, draw, [setpoint])
    has_obs_dim = obs_dim in mu.dims
    # Check if sigma has more than 2 dimensions (chain, draw)
    has_obs_dim = len(sigma.dims) > 2
    has_obs_dim = obs_dim in mu.dims and len(sigma.dims) > 2
    if likehood.NORMAL:
        # shape: (chain, draw)
        if has_obs_dim:
            idata.posterior["var_residual"] = sigma.mean(axis=-1)
        else:
            idata.posterior["var_residual"] = sigma
    elif likehood.T:
        if nu_var is None:
            raise ValueError("For student_t, provide nu_var")
        # If your model is: y ~ StudentT(mu, sigma, nu) => Var = sigma² * nu/(nu-2)
        var_residual = (
            idata.posterior[sigma_var] ** 2
            * idata.posterior[nu_var]
            / (idata.posterior[nu_var] - 2)
        )
        # shape: (chain, draw)
        idata.posterior["var_residual"] = (
            var_residual.mean(dim=obs_dim) if has_obs_dim else var_residual
        )
    return idata


def binomial_band(n, grid, alpha=0.05):
    lower = scipy.stats.binom.ppf(alpha / 2, n, grid) / n
    upper = scipy.stats.binom.ppf(1 - alpha / 2, n, grid) / n
    return lower, upper


def compute_loo_pit_model_agnostic(y_obs, y_pred_flat, weights):
    """Compute model-agnostic LOO Probability Integral Transform (PIT) values.

    This function computes leave-one-out (LOO) PIT values using importance
    sampling weights obtained from PSIS-LOO. It is fully model-agnostic and
    does not assume any specific likelihood (Gaussian, Student-t, etc.).

    Parameters
    ----------
    y_obs : array-like, shape (n_obs,)
        Observed response values.

    y_pred_flat : ndarray, shape (n_obs, n_samples)
        Posterior predictive draws for each observation, flattened across
        chains and draws. Each column corresponds to one observation, and
        each row corresponds to one posterior sample.

    weights : ndarray, shape (n_obs, n_samples)
        PSIS-LOO importance weights for each observation. Each row should
        sum to 1 and corresponds to the normalized importance weights:

            w_i^(s) ∝ 1 / p(y_i | θ^(s))

        approximating p(θ | y_-i).

    Returns
    -------
    loo_pit : ndarray, shape (n_obs,)
        LOO-PIT values for each observation:

            PIT_i = P(y_i^rep ≤ y_i | y_-i)

        computed as a weighted empirical CDF of posterior predictive draws.

    Notes
    -----
    For each observation i, the LOO predictive distribution is approximated as:

        p(y_i | y_-i) ≈ Σ_s w_i^(s) δ(y_i^(s))

    where:
        - y_i^(s) are posterior predictive draws
        - w_i^(s) are PSIS importance weights
        - δ(·) denotes the empirical measure

    The LOO-PIT value is then:

        PIT_i = Σ_s w_i^(s) * 1{ y_i^(s) ≤ y_i }

    Under correct model specification and calibration:

        loo_pit ~ Uniform(0, 1)

    Therefore, deviations from uniformity indicate predictive miscalibration.

    Interpretation
    --------------
    - If loo_pit values are approximately Uniform(0,1),
      the model is well calibrated.
    - Systematic deviations (e.g., S-shaped ECDF) indicate
      under- or over-dispersion.
    - Skewed deviations indicate asymmetric miscalibration.

    Advantages
    ----------
    - Fully model-agnostic
    - Requires no analytic CDF
    - Works for any likelihood
    - Uses PSIS-LOO without refitting the model

    This implementation avoids nonlinear transformations such as HDI-based
    summaries, ensuring stable importance-weighted computation.

    """
    n_obs, n_samples = y_pred_flat.shape
    loo_pit = np.zeros(n_obs)

    for i in range(n_obs):
        indicators = (y_pred_flat[i, :] <= y_obs[i]).astype(float)
        loo_pit[i] = np.sum(weights[i] * indicators)

    return loo_pit


@dataclass
class CalibrationStats:
    expected_coverage: np.ndarray
    empirical_coverage: np.ndarray
    coverage_lower: np.ndarray
    coverage_upper: np.ndarray
    reference_lower: np.ndarray
    reference_upper: np.ndarray
    calibration_error: np.float16
    weighted_cal_error: np.float16
    n_miscalibrated: np.int16


def plot_loo_calibration_curve_with_reference(
    y_obs,
    y_pred,
    weights,
    n_boot=10000,
    ci_level=0.95,
    figsize: tuple[float, float] = (7, 7),
    random_seed: int | None = None,
):
    """Plot calibration curve.

    It uses LOO predictive intervals with Bayesian bootstrap uncertainty.

    Parameters
    ----------
    idata : InferenceData
        ArviZ InferenceData object
    var_name : str
        Name of observed variable
    n_boot : int
        Number of Bayesian bootstrap samples
    ci_level : float
        Confidence level for uncertainty bands
    random_seed : int
        Random seed for reproducibility

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )

    n_obs = y_obs.size

    loo_pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)

    expected_coverage = np.array([*np.arange(0.05, 0.96, 0.05).tolist(), 0.99])
    empirical_coverage = np.array([(loo_pit <= q).mean() for q in expected_coverage])

    # calculate bands
    # finite-sample uncertainty band
    binomial_lower, binomial_upper = binomial_band(len(loo_pit), expected_coverage)
    # posterior uncertainty
    coverage_lower, coverage_upper = bayesian_bootstrap_band(
        loo_pit, expected_coverage, rng, ci_level=ci_level, B=n_boot
    )
    # Compute calibration errors
    calibration_error = np.mean(np.abs(empirical_coverage - expected_coverage))
    print(f"\nMean Calibration Error (LOO): {calibration_error:.3f}")
    # Weighted calibration error
    weights = expected_coverage * (1 - expected_coverage)
    weighted_cal_error = np.average(
        np.abs(empirical_coverage - expected_coverage),
        weights=weights,
    )
    print(f"Weighted Calibration Error (LOO): {weighted_cal_error:.3f}")

    # Statistical test: how many points have CIs that don't contain the diagonal?
    diagonal_values = expected_coverage
    miscalibrated = (diagonal_values < coverage_lower) | (
        diagonal_values > coverage_upper
    )
    n_miscalibrated = cast(np.int16, np.sum(miscalibrated))
    print(
        "Significantly miscalibrated intervals: "
        f"{n_miscalibrated} out of {len(expected_coverage)} "
        f"({100 * n_miscalibrated / len(expected_coverage):.1f}%)"
    )

    # Plot
    _, ax = pyplot.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration", alpha=0.7)

    # REFERENCE band (what we'd expect if well-calibrated)
    ax.fill_between(
        expected_coverage,
        binomial_lower,
        binomial_upper,
        alpha=0.3,
        color="gray",
        label=f"{int(ci_level * 100)}% Expected sampling variation",
    )

    # ACTUAL uncertainty band around empirical coverage
    ax.fill_between(
        expected_coverage,
        coverage_lower,
        coverage_upper,
        alpha=0.3,
        color="steelblue",
        label=f"{int(ci_level * 100)}% Bayesian Bootstrap",
    )

    # Empirical coverage line
    ax.plot(
        expected_coverage,
        empirical_coverage,
        "o-",
        linewidth=2.5,
        markersize=7,
        label="LOO Calibration",
        color="steelblue",
        zorder=10,
    )

    # Highlight significantly miscalibrated points
    if n_miscalibrated > 0:
        ax.scatter(
            expected_coverage[miscalibrated],
            empirical_coverage[miscalibrated],
            s=150,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            zorder=11,
            label="Significantly miscalibrated",
        )

    ax.set_xlabel("Expected Coverage (HDI Level)", fontsize=13)
    ax.set_ylabel("Empirical Coverage (LOO)", fontsize=13)
    ax.set_title(
        f"LOO-Based Calibration Curve (n={n_obs})", fontsize=15, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    # Add sample size info
    ax.text(
        0.98,
        0.02,
        f"n = {n_obs}\nBayesian bootstrap = {n_boot:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
    )

    pyplot.tight_layout()
    pyplot.show()

    return CalibrationStats(
        expected_coverage=expected_coverage,
        empirical_coverage=empirical_coverage,
        coverage_lower=coverage_lower,
        coverage_upper=coverage_upper,
        reference_lower=binomial_lower,
        reference_upper=binomial_upper,
        calibration_error=calibration_error,
        weighted_cal_error=weighted_cal_error,
        n_miscalibrated=n_miscalibrated,
    )


def _marginal_density_single_draw(mu_vec, sigma_scalar, y_g) -> jnp.ndarray:
    """Compute marginal density for one posterior draw."""
    # mu_vec:       (N,)  ← one posterior draw's μ for all observations
    # sigma_scalar: ()    ← one posterior draw's σ
    # y_g:          (N,)  ← the y grid (always the same)
    per_obs = jax_norm.pdf(
        y_g[:, None],  # (G, 1)
        loc=mu_vec[None, :],  # (1, N)
        scale=sigma_scalar,
    )  # (G, N) — density at each grid point for each observation
    # Average over observations → marginal density per sample
    return per_obs.mean(axis=1)  # (G,)


def plot_posterior_predictive(
    idata,
    n_samples_to_plot: int = 1000,
    var_name: str = "y_like",
    figsize: tuple = (10, 5),
    random_seed: int | None = None,
):
    posterior_samples_stacked = extract(
        idata, group="posterior", var_names=["mu", "sigma"], combined=True
    )
    mu_post = posterior_samples_stacked.mu.to_numpy()  # (N, chains * draws)
    sigma_post = posterior_samples_stacked.sigma.to_numpy()  # (chains * draws,)
    y_true = idata.observed_data[var_name].to_numpy()
    n_observations = y_true.size

    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    y_pred = rng.normal(loc=mu_post, scale=sigma_post)

    fig, ax = pyplot.subplots(figsize=figsize)
    # Observed
    n_posterior_samples = sigma_post.size
    ax.hist(
        y_true, bins=50, density=True, alpha=0.5, color="blue", label="Observed data"
    )
    ax.hist(
        y_pred.flatten(),
        bins=50,
        density=True,
        alpha=0.3,
        color="red",
        label="Posterior predictive (pooled)",
    )
    y_grid = np.linspace(y_true.min() - 0.5, y_true.max() + 0.5, 300)
    sample_indices = rng.choice(n_posterior_samples, n_samples_to_plot, replace=False)
    # vmap over posterior samples
    # G, N, S = 300, 130, 1000
    marginal_densities = vmap(
        _marginal_density_single_draw,
        in_axes=(1, 0, None),  # mu: slice along axis 1, sigma: axis 0, y_grid: no map
        out_axes=1,  # output: map over samples
    )(
        jnp.array(mu_post[:, sample_indices]),  # (N, S)
        jnp.array(sigma_post[sample_indices]),  # (S,)
        jnp.array(y_grid),  # (G,)
    )  # (G, S)
    # vmap replaces the for loop below, efficiently computing marginal densities
    # for all samples
    # for s in sample_indices:
    #     # Model-implied marginal density for draw s:
    #     # Average the per-observation normals
    #     # 1/n sum_{i=1}_{i=n} Normal(y | μ_i_{^s}, σ_{^s}) for this draw s
    #     marginal_density = np.mean(
    #         [
    #             scipy.stats.norm.pdf(y_grid, mu_post[i, s], sigma_post[s])
    #             for i in range(len(y_true))
    #         ],
    #         axis=0,
    #     )
    #     ax.plot(
    #         y_grid,
    #         marginal_density,
    #         color="red",
    #         alpha=0.05,
    #     )
    ax.plot(
        y_grid,
        marginal_densities,
        color="red",
        alpha=0.05,
    )
    ax.set_xlabel("Scaled outlet pressure")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend()
    pyplot.show()

    s = rng.choice(n_posterior_samples)
    # Theoretical maximum density at this sample
    fmax = 1 / (sigma_post[s] * np.sqrt(2 * np.pi))

    fig_interactive = go.Figure()

    # Add all N traces, initially visible but togglable
    # 60 * (129 / n_observations) term spaces all N colors perfectly evenly across
    # the full 360° wheel without ever landing on the same hue twice
    colors = [
        f"hsl({int(h)}, 70%, 50%)"
        for h in np.linspace(0, 360 * (129 / n_observations), n_observations)
    ]

    for obs_idx in range(n_observations):
        d = scipy.stats.norm.pdf(y_grid, mu_post[obs_idx, s], sigma_post[s])
        fig_interactive.add_trace(
            go.Scatter(
                x=y_grid,
                y=d,
                mode="lines",
                name=f"Obs {obs_idx}",
                line={"color": colors[obs_idx], "width": 1.5},
                opacity=0.7,
                hovertemplate=f"Obs {obs_idx}: %{{y:.3f}}<extra></extra>",
            )
        )

    # Add a mean curve across all observations
    mean_d = np.mean(
        [
            scipy.stats.norm.pdf(y_grid, mu_post[obs_idx, s], sigma_post[s])
            for obs_idx in range(n_observations)
        ],
        axis=0,
    )

    fig_interactive.add_trace(
        go.Scatter(
            x=y_grid,
            y=mean_d,
            mode="lines",
            name="⟨Mean⟩",
            line={"color": "black", "width": 3, "dash": "dash"},
            opacity=1.0,
            hovertemplate="Mean: %{y:.3f}<extra></extra>",
        )
    )

    fig_interactive.update_layout(
        title={
            "text": (
                f"Posterior Predictive Distributions — Sample {s}"
                f"<br><sup>f<sub>max</sub> = 1 / (σ√2π) = 1 / "
                f"({sigma_post[s]:.3f} × √2π) = {fmax:.3f}</sup>"
            ),
            "font": {"size": 18},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="y",
        yaxis_title="Density",
        width=1700,
        height=550,
        hovermode="closest",
        legend={
            "orientation": "h",  # horizontal legend
            "yanchor": "top",
            "y": -0.15,  # below the plot
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 9},
            "itemwidth": 50,
            "traceorder": "normal",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis={"showgrid": True, "gridcolor": "lightgrey"},
        yaxis={
            "range": [0, np.ceil(fmax * 1.10)],  # 10% headroom above fmax
            "showgrid": True,
            "gridcolor": "lightgrey",
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": 1.5,
                "x": 0.0,
                "xanchor": "center",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Show All",
                        "method": "restyle",
                        "args": [{"visible": True}],
                    },
                    {
                        "label": "Hide All",
                        "method": "restyle",
                        "args": [{"visible": "legendonly"}],
                    },
                ],
            }
        ],
    )

    fig_interactive.show()

    return fig, fig_interactive


def compute_psis_weights(ll):
    """Compute log weights for Pareto-smoothed importance sampling (PSIS) method."""
    # Compute PSIS weights (these reweight posterior samples to approximate LOO
    # posterior)
    log_weights, pareto_k = array_stats.psislw(-ll)
    assert all(pareto_k <= 0.7), (
        "Warning: PSIS Pareto k values indicate potential issues with LOO estimates "
        "(k > 0.7)."
    )
    # log_weights shape (n_obs, n_samples) - log weights for each observation
    # pareto_k shape (n_obs, ) - pareto_k for each observation
    # Normalize weights for each observation considering numerical stability
    weights = np.exp(log_weights - np.max(log_weights, axis=1, keepdims=True))
    weights = weights / np.sum(weights, axis=1, keepdims=True)  # (n_obs, n_samples)

    return weights, pareto_k


def evaluate_model_elpd(
    y_obs,
    elpd_data,
    posterior_sigma,
    ppc_flat,
    weights,
    pareto_k,
    trouble_obs_indices=None,
    well_predicted_indices=None,
    ncols_loo=3,
):
    pointwise_densities = np.exp(elpd_data.elpd_i)
    avg_loo_log_density = elpd_data.elpd / elpd_data.n_data_points
    geom_mean_density = np.exp(avg_loo_log_density)

    analytical_log_density_dist = (
        -0.5 * np.log(2 * np.pi) - 0.5 - np.log(posterior_sigma)
    )
    ideal_mean_density = np.exp(analytical_log_density_dist.mean())
    sigma_model = posterior_sigma.mean()
    deviation = np.sqrt(
        2 * ((-0.5 * np.log(2 * np.pi * sigma_model**2)) - avg_loo_log_density)
    )
    # Auto-detect trouble indices if not provided
    if trouble_obs_indices is None:
        trouble_obs_indices = np.where(
            pointwise_densities
            < (pointwise_densities.mean() - 2 * pointwise_densities.std())
        )[0]

    if well_predicted_indices is None:
        well_predicted_indices = np.argsort(pointwise_densities.values)[-5:]

    print(
        "Poorly predicted observations (abnormally low densities): "
        f"{trouble_obs_indices}"
    )
    print(
        "The top 5 predicted observations (highest densities): "
        f"{well_predicted_indices}"
    )

    obs_to_plot = np.concatenate([trouble_obs_indices, well_predicted_indices])
    nrows_loo = int(np.ceil(len(obs_to_plot) / ncols_loo))

    # Outer grid: row 0 = first two plots, row 1 = LOO grid
    fig = pyplot.figure(figsize=(25, 8 + 4 * nrows_loo))
    outer_gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, nrows_loo], hspace=0.5
    )

    # First row: two side-by-side plots
    inner_gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[0], wspace=0.3
    )
    ax1 = fig.add_subplot(inner_gs_top[0])
    ax2 = fig.add_subplot(inner_gs_top[1])

    # Second row: LOO predictive grid
    inner_gs_loo = gridspec.GridSpecFromSubplotSpec(
        nrows_loo, ncols_loo, subplot_spec=outer_gs[1], wspace=0.3, hspace=0.1
    )
    loo_axes = [fig.add_subplot(inner_gs_loo[i]) for i in range(len(obs_to_plot))]

    # --- Plot 1: histogram ---
    ax1.hist(analytical_log_density_dist)
    ax1.axvline(
        analytical_log_density_dist.mean(),
        color="black",
        label="expected log-density under perfect calibration",
    )
    ax1.axvline(
        avg_loo_log_density,
        color="red",
        label="actual average out-of-sample predictive performance",
    )
    ax1.set_title(
        "Distribution of the ideal log-density across posterior draws of σ \n"
        f"root-mean-square standardized residual {deviation:.3f}",
        fontsize=14,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        framealpha=0.4,
        columnspacing=0.8,
        handlelength=1.2,
    )

    # --- Plot 2: scatter ---
    ax2.scatter(y_obs, pointwise_densities)
    ax2.axhline(
        geom_mean_density,
        color="red",
        linestyle="--",
        label=f"Geometric mean density = {geom_mean_density:.2f}",
    )
    ax2.axhline(
        ideal_mean_density,
        color="black",
        linestyle="--",
        label=f"expected density under perfect calibration = {ideal_mean_density:.2f}",
    )
    ax2.set_xlabel("Observed Value")
    ax2.set_ylabel("Pointwise ELPD contribution")
    ax2.set_title("Pointwise ELPD contributions for each observation")
    ax2.set_yscale("log")
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        framealpha=0.4,
        columnspacing=0.8,
        handlelength=1.2,
    )

    # --- Plot 3: LOO predictive grid ---
    for ax, i in zip(loo_axes, obs_to_plot, strict=False):
        plot_loo_predictive_single(
            i, y_obs, ppc_flat, weights, pareto_k, pointwise_densities, ax=ax
        )
    fig.suptitle("Model ELPD Evaluation", fontsize=16, y=1.01)

    pyplot.show()

    return trouble_obs_indices


def plot_loo_predictive_single(
    i, y_obs, ppc_flat, weights, pareto_k, pointwise_densities, ax
):
    w = weights[i, :]
    samples = ppc_flat[i, :]

    loo_mean = np.average(samples, weights=w)
    loo_std = np.sqrt(np.average((samples - loo_mean) ** 2, weights=w))
    z_score = (y_obs[i] - loo_mean) / loo_std

    kde = scipy.stats.gaussian_kde(samples, weights=w)
    x = np.linspace(samples.min(), samples.max(), 300)

    loo_density_at_obs_psis = pointwise_densities[i]
    loo_density_at_obs_kde = kde(y_obs[i])[0]

    ax.plot(x, kde(x), color="steelblue", lw=2, label="LOO predictive")
    ax.axvline(
        y_obs[i], color="red", linestyle="--", lw=2, label=f"Observed = {y_obs[i]:.2f}"
    )
    ax.axvline(
        loo_mean,
        color="steelblue",
        linestyle=":",
        lw=1.5,
        label=f"LOO mean = {loo_mean:.2f}",
    )

    x_shade = np.linspace(loo_mean - loo_std, loo_mean + loo_std, 300)
    ax.fill_between(x_shade, kde(x_shade), alpha=0.15, color="steelblue", label="±1 SD")

    ax.scatter(
        [y_obs[i]],
        [loo_density_at_obs_psis],
        color="red",
        zorder=5,
        label=f"PSIS density = {loo_density_at_obs_psis:.2f}",
    )
    ax.scatter(
        [y_obs[i]],
        [loo_density_at_obs_kde],
        color="orange",
        zorder=5,
        marker="x",
        s=80,
        label=f"KDE density = {loo_density_at_obs_kde:.2f}",
    )

    ax.set_title(
        f"Obs {i} | k={pareto_k[i]:.2f} | z={z_score:.2f}",
        color="red" if abs(z_score) > 1 else "black",
    )
    ax.set_xlabel("y")
    ax.set_ylabel("LOO predictive density")
    ax.legend(fontsize=8)


def visualize_data(
    y_obs: np.ndarray, bound_width: int | None = None, hist_bins: int | None = None
):
    kde = scipy.stats.gaussian_kde(y_obs, bw_method=bound_width)
    y_grid = np.linspace(y_obs.min() - 0.5, y_obs.max() + 0.5, 300)
    kde_densities = kde.evaluate(y_grid)
    peaks, _ = scipy.signal.find_peaks(kde_densities, distance=25)
    centers_kde = y_grid[peaks].round(3)
    # find center of each cluster (representing each setpoint)
    # GMM is more appropriate when clusters have different widths and densities
    gmm = GaussianMixture(n_components=4, random_state=14)
    gmm.fit(y_obs.reshape(-1, 1))
    centers_gmm = sorted(gmm.means_.round(3).flatten().tolist())
    fig, axs = pyplot.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(
        y_obs,
        bins=hist_bins,
        density=True,  # density=True normalizes the histogram to represent a probability
        # density
        # weights=np.ones_like(y_obs)
        # / len(y_obs),  # turns on when density=False to represent probability
    )
    axs[0].plot(y_grid, kde_densities, label="kde", color="red")
    print("Cluster centers through KDE peaks:", centers_kde)
    print("Cluster centers through GMM method:", centers_gmm)
    for c in centers_kde:
        axs[0].axvline(c, color="r", ls="--")
    axs[0].set_title("Distribution of the scaled outlet pressure")
    axs[0].set_xlabel("Scaled outlet pressure")
    axs[0].set_ylabel("Density")
    axs[1].hist(y_obs, bins=hist_bins, density=True, cumulative=True, histtype="step")
    axs[1].set_title("Empirical Cumulative Distribution Function (CDF)")
    axs[1].set_xlabel("Data Values")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].grid(True)
    axs[0].legend()
    pyplot.tight_layout()
    pyplot.show()
    return fig


def compute_autocorr(x):
    """Vectorized lag-1 autocorrelation across rows (shape: n_draws x n_timesteps)."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    # Lag-1: correlate x[t] with x[t-1] across all draws at once
    num = (x_centered[:, :-1] * x_centered[:, 1:]).sum(axis=1)
    denom = np.sqrt(
        (x_centered[:, :-1] ** 2).sum(axis=1) * (x_centered[:, 1:] ** 2).sum(axis=1)
    )
    return num / denom


def get_mixture_residuals(idata, y):
    """Assign observations and componenets.

    Assign each observation to its most likely mixture component.
    and compute residuals relative to that component.

    Parameters
    ----------
    idata : arviz.InferenceData
        Fitted mixture model inference data
    y : np.ndarray
        Observed data (standardized deviations)

    Returns
    -------
    dict with:
        - residuals: y - predicted_mu (raw residuals)
        - standardized_residuals: (y - predicted_mu) / predicted_sigma
        - assignment: which component (0 or 1) each observation belongs to
        - probability: soft assignment probabilities [P(k=0), P(k=1)]
        - predicted_mu: the mean of assigned component for each obs
        - predicted_sigma: the sigma of assigned component for each obs

    """
    # Extract posterior means of mixture parameters
    weights = idata.posterior["weights"].mean(dim=["chain", "draw"]).to_numpy()
    mu = idata.posterior["mu"].mean(dim=["chain", "draw"]).to_numpy()
    sigma = idata.posterior["sigma"].mean(dim=["chain", "draw"]).to_numpy()

    print("Learned mixture parameters:")
    print(f"  weights: {weights}")
    print(f"  mu:      {mu}")
    print(f"  sigma:   {sigma}")

    # Compute log-responsibilities for each observation
    # P(component k | y_i) ∝ P(y_i | component k) * P(component k)
    n_obs = len(y)
    n_components = len(mu)

    log_prob = np.zeros((n_obs, n_components))
    for k in range(n_components):
        # log P(y_i | k) + log P(k)
        log_prob[:, k] = np.log(weights[k]) + scipy.stats.norm.logpdf(
            y, mu[k], sigma[k]
        )

    # Convert to probabilities via softmax
    log_prob_max = log_prob.max(axis=1, keepdims=True)
    prob = np.exp(log_prob - log_prob_max)
    prob /= prob.sum(axis=1, keepdims=True)

    # Hard assignment: assign to most likely component
    # Observations near the boundary between components will have large residuals
    # under either component's mean
    assignment = np.argmax(prob, axis=1)

    # Compute residuals relative to assigned component
    predicted_mu = mu[assignment]
    predicted_sigma = sigma[assignment]
    residuals = y - predicted_mu
    standardized_residuals = residuals / predicted_sigma

    return {
        "residuals": residuals,
        "standardized_residuals": standardized_residuals,
        "assignment": assignment,
        "probability": prob,
        "predicted_mu": predicted_mu,
        "predicted_sigma": predicted_sigma,
    }


def plot_mixture_residuals(residuals_dict):
    """Plot residuals from mixture model, colored by assigned component and setpoint.

    Parameters
    ----------
    residuals_dict : dict
        Output from get_mixture_residuals containing residuals, assignments, etc.
    sp_idx : array-like
        Setpoint index for each observation

    Returns
    -------
    fig : matplotlib Figure
        Residual plot figure

    """
    standardized_residuals = residuals_dict["standardized_residuals"]
    assignment = residuals_dict["assignment"]

    # Plot residual diagnostics
    fig, axes = pyplot.subplots(1, 2, figsize=(15, 4))

    # 1. Q-Q plot (should follow diagonal)

    scipy.stats.probplot(
        standardized_residuals, dist="norm", plot=axes[0], fit=True, rvalue=True
    )
    axes[0].set_title("Q-Q Plot")

    # 2. Assignment counts
    n_low = (assignment == 0).sum()
    n_high = (assignment == 1).sum()
    axes[1].bar(
        ["Low Mode", "High Mode"],
        [n_low, n_high],
        alpha=0.7,
        edgecolor="black",
        color=["blue", "orange"],
    )
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Mode Assignment\n(Low: {n_low}, High: {n_high})")

    pyplot.tight_layout()
    pyplot.show()
    return fig


def build_mixture_baseline(
    y_deviation,
    kwargs: dict,
    n_components=2,
    random_seed: int | None = None,
):
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )

    """Build a mixture model for the residuals."""
    coords = {
        "obs_id": np.arange(len(y_deviation)),
        "component": np.arange(n_components),
    }

    with pm.Model(coords=coords) as model:
        # Data
        y_data = pm.Data("y_data", y_deviation, dims="obs_id")
        # ---------
        # Mixture weights
        # ---------
        weights = pm.Dirichlet(
            "weights",
            a=np.ones(n_components),
            shape=(n_components,),
            dims="component",
            rng=rng,
        )
        # ---------
        # Ordered component means (prevents label switching)
        # ---------
        mu_unordered = pm.Normal(
            "mu_unordered",
            mu=kwargs["mu_mu"],
            sigma=kwargs["mu_sigma"],
            shape=n_components,
            dims="component",
            rng=rng,
        )
        mu = pm.Deterministic("mu", pt.sort(mu_unordered), dims="component")
        # ---------
        # Component standard deviations
        # ---------
        sigma = pm.HalfNormal(
            "sigma",
            sigma=kwargs["sigma_sigma"],
            shape=n_components,
            dims="component",
            rng=rng,
        )
        # ---------
        # Component distributions
        # ---------
        componets = pm.Normal.dist(mu=mu, sigma=sigma, shape=(n_components,))
        # ---------
        # Mixture likelihood
        # ---------
        pm.Mixture(
            "y_like",
            w=weights,
            comp_dists=componets,
            observed=y_data,
            dims="obs_id",
            rng=rng,
        )

    return model


def evaluate_mixture_model(idata, y_true, y_grid: np.ndarray):
    pdf_components = (
        XrContinuousRV(
            scipy.stats.norm, idata.posterior["mu"], idata.posterior["sigma"]
        ).pdf(y_grid)
        * idata.posterior["weights"]
    )
    pdf = pdf_components.sum("component")
    fig, ax = pyplot.subplots(3, 1, figsize=(7, 8), sharex=True)
    # empirical histogram
    ax[0].hist(y_true, density=True, bins=25)
    # pdf of the fitted mixture model
    pdf_components.mean(dim=["chain", "draw"]).sum("component").plot.line(ax=ax[1])
    ax[1].set(title="PDF", xlabel="y", ylabel="Probability\ndensity")
    # plot group membership probabilities
    (pdf_components / pdf).mean(dim=["chain", "draw"]).plot.line(
        hue="component", ax=ax[2]
    )
    ax[2].set(title="Group membership", xlabel="y", ylabel="Probability")
    pyplot.tight_layout()
    return fig
