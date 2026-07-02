"""Utility functions for compressor fouling modeling analysis.

This module provides tools for data preparation, model training, SHAP analysis,
residual analysis, CUSUM-based anomaly detection, and Bayesian regression
evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import io
from itertools import groupby
from typing import Any, Literal, TypedDict, cast
import warnings

from arviz_base import extract
import arviz_plots as azp
from arviz_plots.plot_collection import PlotCollection
from arviz_stats import ELPDData
from arviz_stats.base.array import array_stats
from arviz_stats.loo import compare
from arviz_stats.loo.loo_expectations import loo_metrics, loo_r2
from arviz_stats.metrics import bayesian_r2, metrics as azs_metrics, residual_r2
from arviz_stats.visualization import hdi
from IPython.display import display
from jax import Array, vmap
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
from jaxtyping import ScalarLike
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from numpy._core import uint16
from numpy._typing import NDArray
import optuna
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
from pymc.initial_point import StartDict
from pymc.variational.opvi import DataArray
import pytensor.tensor as pt
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.variable import TensorVariable
import scipy
from scipy import stats
from sklearn import linear_model
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LearningCurveDisplay, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
import xarray as xr
from xarray import DataTree
from xarray_einstats.stats import XrContinuousRV

Float64Matrix1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Float64Matrix2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Uint16Matrix1D = np.ndarray[tuple[int], np.dtype[np.uint16]]
BoolMatrix1D = np.ndarray[tuple[int], np.dtype[np.bool_]]


class ElasticNetParams(TypedDict):
    alpha: float
    l1_ratio: float
    fit_intercept: bool


class SplineTransformerParams(TypedDict):
    n_knots: int
    knots: str
    degree: int
    include_bias: bool
    extrapolation: str


class HierarchicalModelParams(TypedDict):
    setpoint_unique: Float64Matrix1D
    setpoint_index: Uint16Matrix1D
    sigma_sd_sd: float


class CalibrationCurveParams(TypedDict):
    expected_coverage: Float64Matrix1D
    empirical_coverage: Float64Matrix1D
    sampling_lower: Float64Matrix1D
    sampling_upper: Float64Matrix1D
    bootstrap_lower: Float64Matrix1D
    bootstrap_upper: Float64Matrix1D
    calibration_error: np.float64
    weighted_cal_error: np.float64
    miscalibrated: BoolMatrix1D
    n_miscalibrated: np.uint16
    n_obs: int


@dataclass
class PerformanceResult:
    """Overall model performance on the original (unstandardized) scale."""

    mu_hat: Float64Matrix1D
    y_original: Float64Matrix1D
    mu_hat_original: Float64Matrix1D
    r2_overall: float


@dataclass
class VarDecompResult:
    """Variance decomposition into between- and within-setpoint components."""

    var_total: float
    var_between: float
    var_within: float


@dataclass
class PluginResult:
    """Plug-in residual sigma, its bootstrap CI, and within-setpoint R²."""

    plugin_sigma: dict[np.uint16, float]
    plugin_sigma_ci: dict[np.uint16, np.ndarray]
    within_setpoint_r2: dict[np.uint16, float]


@dataclass(frozen=True, slots=True)
class UncertainCusumResult:
    cusum_paths: Float64Matrix2D  # (n_samples, n_obs)
    mean: Float64Matrix1D  # (n_obs)
    median: Float64Matrix1D  # (n_obs)
    std: Float64Matrix1D  # (n_obs)
    ci_95: Float64Matrix2D  # (2, n_obs)
    ci_80: Float64Matrix2D  # (2, n_obs)
    ci_50: Float64Matrix2D  # (2, n_obs)
    timestamps: pd.DatetimeIndex  # (n_obs)
    target_means: Float64Matrix1D  # (n_samples,)
    drifts: Float64Matrix1D  # (n_samples,)
    anomaly_direction: Literal["neg", "pos"] = "neg"


@dataclass(frozen=True, slots=True)
class UncertainResidualStats:
    mean: Float64Matrix1D  # (n_obs,)
    median: Float64Matrix1D  # (n_obs,)
    ci_95: Float64Matrix2D  # (2, n_obs)
    ci_80: Float64Matrix2D  # (2, n_obs)
    ci_50: Float64Matrix2D  # (2, n_obs)
    timestamps: pd.DatetimeIndex  # (n_obs,)
    operational_mask: BoolMatrix1D  # (n_obs,)


@dataclass(frozen=True, slots=True)
class MixtureModelResidualStats:
    residuals: Float64Matrix1D  # (n_obs,)
    standardized_residuals: Float64Matrix1D  # (n_obs,)
    assignment: Float64Matrix1D  # (n_obs,)
    probability: Float64Matrix2D  # (n_obs, n_comp)
    predicted_mu: Float64Matrix1D  # (n_obs,)
    predicted_sigma: Float64Matrix1D  # (n_obs,)


def _prepare_baseline_mask(
    data: pd.DataFrame,
    baseline_period: list[tuple[str, str]] | None = None,
    shutin_mask: pd.Series | None = None,
    n_init_samples: int = 60,
    min_periods: int = 3,
    rolling_mean_std_multiplier: float = 1.25,
) -> pd.Series:
    """Prepare a baseline mask for data analysis.

    Creates a boolean mask to identify data points within the baseline period,
    using either a predefined baseline period or calculated from the initial
    samples of the data. The mask is adjusted based on a rolling mean and
    standard deviation. It assumes data index is DatetimeIndex and that
    'Outlet_Pressure_SP' column is present in the DataFrame and contains the
    outlet pressure setpoint values.

    Args:
        data: DataFrame with timestamp index containing the input data.
            It should contain 'Outlet_Pressure_SP' column.
        baseline_period: List of tuples representing start and end times for the
            baseline period. Defaults to None.
        shutin_mask: Boolean Series indicating periods when the system was
            shut in. Defaults to None.
        n_init_samples: Number of initial samples used to calculate the baseline
            mean and std. Defaults to 60.
        min_periods: Minimum number of periods required for rolling statistics
            calculation. Defaults to 3.
        rolling_mean_std_multiplier: Multiplier for baseline standard deviation
            to determine the threshold for masking. Defaults to 1.25.

    Returns:
        Boolean Series representing the baseline mask.

    """
    # Initialize the baseline mask as a boolean series of False values
    # with the same index as the input DataFrame.
    baseline_mask = pd.Series(False, index=data.index)

    if baseline_period is None:
        data["outlet_pressue_sp_track"] = (
            data["Outlet_Pressure"] - data["Outlet_Pressure_SP"]
        ) / (data["Outlet_Pressure_SP"] + 1e-6)
        baseline_mean: float = (
            data["outlet_pressue_sp_track"].iloc[:n_init_samples].mean()
        )
        baseline_std: float = (
            data["outlet_pressue_sp_track"].iloc[:n_init_samples].std()
        )
        if shutin_mask is not None:
            data["outlet_pressue_sp_track_rolled_mean"] = (
                data.loc[:, "outlet_pressue_sp_track"]
                .mask(shutin_mask)
                .rolling(window=min_periods, center=False)
                .mean()
                .bfill()
                .where(~shutin_mask)
            )
        else:
            data["outlet_pressue_sp_track_rolled_mean"] = (
                data.loc[:, "outlet_pressue_sp_track"]
                .rolling(window=min_periods, center=False)
                .mean()
                .bfill()
            )
        baseline_mask: pd.Series = (
            data["outlet_pressue_sp_track_rolled_mean"]
            > baseline_mean - rolling_mean_std_multiplier * baseline_std
        )
    if baseline_period is not None:
        for start, end in baseline_period:
            baseline_mask |= (data.index >= start) & (data.index < end)
    return baseline_mask


def _prepare_shutin_mask(data: pd.DataFrame) -> pd.Series:
    """Prepare a mask indicating periods when the system was shut in.

    Creates a mask that identifies periods when the system was shut in based on
    the outlet pressure setpoint being zero. It assumes that 'outlet_pressue_sp'
    column is present in the DataFrame and contains the outlet pressure setpoint
    values.


    Args:
        data (pd.DataFrame): DataFrame with timestamp index containing
            'Outlet_Pressure_SP' column.

    Returns:
        Boolean series where True indicates periods when the system was shut in
        based on outlet pressure setpoint.

    """
    return (data["Outlet_Pressure_SP"] == 0) & data["Outlet_Pressure_SP"].notna()


def calculate_data_masks(
    data: pd.DataFrame,
    baseline_period: list[tuple[str, str]] | None = None,
    n_init_samples: int = 60,
    min_periods: int = 3,
    rolling_mean_std_multiplier: float = 1.25,
) -> tuple[pd.Series, pd.Series]:
    """Calculate baseline and shut-in masks for data analysis.

    Computes two boolean masks from the input data: one for identifying the
    baseline period and another for indicating periods when the system was shut
    in based on the outlet pressure setpoint.

    Args:
        data: DataFrame with timestamp index containing the input data. It
            should contain 'Outlet_Pressure_SP' column.
        baseline_period: List of tuples representing start and end times for the
            baseline period. Defaults to None.
        n_init_samples: Number of initial samples used to calculate the baseline
        mean and std. Defaults to 60.
        min_periods: Minimum number of periods required for rolling statistics
            calculation. Defaults to 3.
        rolling_mean_std_multiplier: Multiplier for baseline standard deviation
        to determine the threshold for masking. Defaults to 1.25.

    Returns:
        Tuple containing two boolean Series: the baseline mask and the shut-in
        mask.

    """
    assert isinstance(data.index, pd.DatetimeIndex), (
        "data index must be a DatetimeIndex."
    )
    assert "Outlet_Pressure_SP" in data.columns, (
        "Outlet_Pressure_SP column not found in the data."
    )

    shutin_mask: pd.Series = _prepare_shutin_mask(data.copy())
    assert shutin_mask.dtype == bool, "shutin_mask must be a boolean Series."

    baseline_mask: pd.Series = _prepare_baseline_mask(
        data.copy(),
        baseline_period,
        shutin_mask,
        n_init_samples,
        min_periods,
        rolling_mean_std_multiplier,
    )
    assert baseline_mask.dtype == bool, "baseline_mask must be a boolean Series."

    return baseline_mask, shutin_mask


def plot_timeseries_grid(
    data: pd.DataFrame,
    config: list[list[tuple[str, str]]],
    figsize: tuple[int, int] = (30, 25),
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot a grid of time series data for multiple configurations.

    Args:
        data: The DataFrame containing the time series data with datetime index.
        config: A configuration where each inner list contains tuples of
            (column_name, color) specifying which columns to plot and their
            respective colors
        figsize: The size of the figure (width, height), default is (30, 25).
        save: Whether to save the plot.
        fname: The filename to save the plot as.

    Raises:
        ValueError: If save is True and fname is None.

    Example:
        ```python
        data = pd.read_csv(
            "time_series_data.csv", parse_dates=["date"], index_col="date"
        )
        config = [
            [
                ("column1", "blue"),
                ("column2", "red"),
                ("column3", "green"),
                ("column4", "purple"),
            ],
        ]
        plot_timeseries_grid(data, config)
        ```
        This will create a grid of plots with four rows and one column.
        Each row represents a configuration from the 'config' list.
        The columns in each row are specified by the tuples within 'config'.
        The plots are colored according to the specified colors.
        The plot is displayed and can be saved to a file using the 'save' parameter.

    """
    fig, axs = plt.subplots(len(config), 1, sharex=True, figsize=figsize)

    for idx, series in enumerate(config):
        for col, color in series:
            axs[idx].plot(data.index, data[col], "o-", color=color, label=col)

        axs[idx].legend()

    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, axs


def prepare_model_input(
    data: pd.DataFrame,
    feature_names: list[str] | None,
    feature_engineering_allowed: bool = True,
    target_name: str = "Outlet_Pressure",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Prepare the input for a machine learning model.

    This function drops missing values, performs interpolation and feature
    engineering.

    Args:
        data: The input DataFrame containing all necessary columns.
        feature_names: A list of column names to be used as features. If None,
            base features will be used.
        feature_engineering_allowed: Whether to allow feature engineering.
            Defaults to True.
        target_name: The name of the target column. Defaults to
        "Outlet_Pressure".

    Returns:
        data: The preprocessed DataFrame.
        X: The feature DataFrame.
        y: The target Series.

    Example:
        ```python
        df = pd.DataFrame(
            {
                "Outlet_Pressure": [10, 20, np.nan],
                "Inlet_Temperature": [30, 40, 50],
                "Outlet_Temperature": [60, 70, 80],
                "Inlet_Flow_Rate": [90, 100, 110],
            }
        )
        prepare_model_input(df, ["Outlet_Pressure", "Inlet_Temperature"])
        (processed_df, X_features, y_target)
        ```

    """
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
        feature_names = base_features
    # If feature_names are provided, use them.
    # Add engineered_features only if feature_engineering_allowed is True.
    if feature_engineering_allowed:
        feature_names += engineered_features

    X = pd.DataFrame(data[feature_names])
    y = data[target_name]
    assert isinstance(y, pd.Series), "y must be a pandas Series."
    return data, X, y


def visualize_correlations(
    ax: Axes,
    data: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    aspect: float | Literal["auto", "equal"] = "equal",
):
    """Visualize the correlation between two features in a DataFrame.

    Args:
        ax: Matplotlib Axes object to plot on.
        data: The DataFrame containing the data.
        feature_1: Name of the first feature.
        feature_2: Name of the second feature.
        aspect: Aspect ratio for the plot. Default is "equal".

    """
    _ = ax.scatter(data[feature_1], data[feature_2])
    _ = ax.set_xlabel(feature_1)
    _ = ax.set_ylabel(feature_2)
    _ = ax.set_aspect(aspect)


def visualize_imputation(ax: Axes, data: pd.DataFrame, feature: str):
    """Visualizes histograms of original and imputed data using different methods.

    Args:
        ax: Matplotlib axes object to plot the histogram.
        data: Pandas DataFrame containing the dataset.
        feature: Name of the feature column to visualize.

    Example:
        ```python
        fig, ax = plt.subplots()
        visualize_imputation(ax, df, "temperature")
        plt.show()
        ```

    """
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
    _ = ax.legend()
    _ = ax.set_title(f"Imputation Methods -\n {feature}")


def visualize_data_folds(
    X: pd.DataFrame,
    y: pd.Series,
    data_folds: list[tuple[Uint16Matrix1D, Uint16Matrix1D]],
    save: bool = False,
    fname: str | None = None,
):
    """Visualizes the distribution of target autocorrelations.

    Args:
        X: Features dataframe.
        y: Target series.
        data_folds: List of tuples containing train and test split indices.
        save: Whether to save the plot. Defaults to False.
        fname: File name to save the plot if save is True.

    Raises:
            ValueError: If save is True and fname is None.

    """
    n_col = len(data_folds)
    fig, axes = plt.subplots(1, n_col, figsize=(20, 7))
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
            + f"train {y_train.autocorr():.3f} test: {y_test.autocorr():.3f}"
        )
        axes[i].legend()

    _ = fig.suptitle("Distribution of target autocorrelations")
    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, axes


def _suggest_lr_params(trial: optuna.Trial) -> ElasticNetParams:
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 100, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 1.0, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }


def _suggest_spline_params(trial: optuna.Trial) -> SplineTransformerParams:
    return {
        "n_knots": trial.suggest_int("n_knots", 2, 4),
        "knots": trial.suggest_categorical("knots", ["uniform", "quantile"]),
        "degree": trial.suggest_int("degree", 1, 3),
        "include_bias": trial.suggest_categorical("include_bias", [True, False]),
        "extrapolation": trial.suggest_categorical(
            "extrapolation", ["constant", "linear", "continue"]
        ),
    }


def _build_lr_pipeline(params: ElasticNetParams) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                linear_model.ElasticNet(
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                    fit_intercept=params["fit_intercept"],
                ),
            ),
        ]
    )


def _build_spline_pipeline(
    elasticnet_params: ElasticNetParams,
    spline_params: SplineTransformerParams,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "spline",
                SplineTransformer(
                    n_knots=spline_params["n_knots"],
                    knots=spline_params["knots"],
                    degree=spline_params["degree"],
                    include_bias=spline_params["include_bias"],
                    extrapolation=spline_params["extrapolation"],
                ),
            ),
            (
                "regressor",
                linear_model.ElasticNet(
                    alpha=elasticnet_params["alpha"],
                    l1_ratio=elasticnet_params["l1_ratio"],
                    fit_intercept=elasticnet_params["fit_intercept"],
                ),
            ),
        ]
    )


def train_model(
    X_baseline: pd.DataFrame,
    y_baseline: pd.Series,
    cv: list[tuple[Uint16Matrix1D, Uint16Matrix1D]],
    model_type: Literal["lr", "spline"] = "lr",
    n_trials: int = 100,
) -> Pipeline:
    """Trains a regression model using Optuna for hyperparameter optimization.

    Sets up and executes a machine learning pipeline with either a linear or
    spline-based regression model. It uses Optuna to perform hyperparameter
    tuning through a specified number of trials, and cross-validation is handled
    by the provided `cv` strategy. After optimizing the model, it prints out the
    best trial's value and parameters, fits the model to the baseline data,
    evaluates its performance using various metrics, and returns the best fitted
    model.

    Args:
        X_baseline: The feature dataset used for training.
        y_baseline: The target variable corresponding to the features.
        cv: Cross-validation strategy object to be passed to cross_val_score.
        model_type: Type of regression model. Defaults to 'lr' (linear).
        n_trials: Number of trials for hyperparameter optimization. Defaults to 100.

    Returns:
        The best fitted regression model after hyperparameter optimization and
        training.

    Note:
        Internally uses ``optuna.TrialPruned`` during optimization to prune
        unpromising trials via ``MedianPruner``.

    Example:
        ```python
        train_model(
            X_train,
            y_train,
            cv=[(train_index, test_index), ...],
            model_type="spline",
        )
        ```

    """

    def _objective(trial: optuna.Trial) -> float:
        if model_type == "lr":
            pipeline = _build_lr_pipeline(_suggest_lr_params(trial))
        else:
            pipeline = _build_spline_pipeline(
                _suggest_lr_params(trial),
                _suggest_spline_params(trial),
            )

        scores = cross_val_score(
            pipeline,
            X_baseline,
            y_baseline,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        trial.report(scores.mean(), step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()
        return scores.mean()

    # Create and optimize study
    study = optuna.create_study(
        direction="maximize",
        study_name="compressor_outlet_pressure",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(_objective, n_trials=n_trials)

    trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Build and train final model
    lr_params = cast(
        ElasticNetParams,
        cast(
            "object",
            {k: trial.params[k] for k in ("alpha", "l1_ratio", "fit_intercept")},
        ),
    )

    if model_type == "lr":
        best_model = _build_lr_pipeline(lr_params)
    else:
        spline_params = cast(
            SplineTransformerParams,
            cast(
                "object",
                {
                    k: trial.params[k]
                    for k in (
                        "n_knots",
                        "knots",
                        "degree",
                        "include_bias",
                        "extrapolation",
                    )
                },
            ),
        )
        best_model = _build_spline_pipeline(lr_params, spline_params)
    _ = best_model.fit(X_baseline, y_baseline)

    # evaluate model on training data
    y_baseline_pred: Float64Matrix1D = best_model.predict(X_baseline)
    baseline_residuals: Float64Matrix1D = y_baseline.to_numpy() - y_baseline_pred
    mae = mean_absolute_error(y_baseline, y_baseline_pred)
    r2 = r2_score(y_baseline, y_baseline_pred)
    n = len(y_baseline)
    p: int = best_model.named_steps["regressor"].coef_.size
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print(f"\nBaselibe Training MAE: {mae:.4f}")
    print(f"Baseline Training R^2: {r2:.4f}")
    print(f"Baseline Training adjusted R^2: {adjusted_r2:.4f}")
    print(f"Baseline Residuals Mean: {baseline_residuals.mean():.3f}")
    print(f"Baseline Residuals Std: {baseline_residuals.std():.3f}")

    return best_model


def visualize_learning_curve(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray,
    cv: list[tuple[np.ndarray, np.ndarray]],
    scoring: str,
    negate_score: bool,
    random_seed: int,
    *,
    shuffle: bool = True,
    save: bool = False,
    fname: str | None = None,
):
    """Visualizes the learning curve for a given estimator.

    Args:
        estimator: The model to evaluate.
        X: Feature matrix.
        y: Target variable.
        train_sizes: Array of training set sizes.
        cv: Cross-validation strategy.
        scoring: Scoring method.
        negate_score: Whether to negate the score.
        random_seed: Random seed for reproducibility.
        shuffle: Whether to shuffle the data before splitting. Defaults to True.
        save: If True, saves the plot to a file. Defaults to False.
        fname: Filename to save the plot if `save` is True. Defaults to None.

    Raises:
        ValueError: If save is True and fname is None.

    """
    fig, ax = plt.subplots(1, 1)
    _ = LearningCurveDisplay.from_estimator(
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
    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, ax


def visualize_partial_dependency(
    estimator: Pipeline,
    X: pd.DataFrame,
    random_seed: int,
    model_type: Literal["lr", "spline"] = "lr",
):
    """Plot partial dependence for each feature in the feature matrix.

    For linear models, features are ordered by absolute coefficient magnitude
    and each subplot is annotated with its coefficient.

    Args:
        estimator: Fitted pipeline whose ``regressor`` step has a
            ``coef_`` attribute.
        X: Feature matrix used to compute the partial dependence.
        random_seed: Seed passed to
            ``PartialDependenceDisplay.from_estimator``.
        model_type: Whether the regressor is a linear model (``"lr"``) or a
            spline-based model (``"spline"``). Defaults to ``"lr"``.

    Example:
        ```python
         from sklearn.pipeline import Pipeline
         from sklearn.linear_model import LinearRegression
         import pandas as pd
         pipe = Pipeline([("regressor", LinearRegression())])
         X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
         pipe.fit(X, [1, 2, 3])
         visualize_partial_dependency(pipe, X, random_seed=42)
         ```

    """
    feature_names = X.columns.tolist()
    coefs = estimator.named_steps["regressor"].coef_
    if model_type == "lr":
        order = np.argsort(np.abs(coefs))[::-1]
        feature_names = X.columns[order].tolist()
        coefs = coefs[order]

    fig, ax = plt.subplots(figsize=(25, 15))
    disp = PartialDependenceDisplay.from_estimator(
        estimator,
        X=X,
        features=feature_names,
        feature_names=feature_names,
        kind="both",
        random_state=random_seed,
        n_jobs=-1,
        ax=ax,
    )
    if model_type == "lr":
        # Add coefficient to each subplot
        for i, (_, coef) in enumerate(zip(feature_names, coefs, strict=False)):
            this_ax = disp.axes_.ravel()[i]
            this_ax.text(
                0.05,
                0.95,
                f"β = {coef:.3f}",
                transform=this_ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
            )
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    del fig, ax


def calculate_residuals(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series
) -> pd.Series:
    """Calculate residuals from a model and data.

    Calculates residuals from a model and data. It assumes that the model has a
    predict method and target vector is a pandas Series with DatetimeIndex.

    Args:
        estimator : trained model object with predict method
        X : feature matrix
        y : target vector

    Returns:
        Series of residuals

    """
    assert isinstance(y.index, pd.DatetimeIndex), "y must have a DatetimeIndex"

    y_pred = estimator.predict(X)
    residuals = y.to_numpy() - y_pred
    return pd.Series(data=residuals, index=y.index)


def find_anomaly_onsets(
    indices: np.ndarray | list[np.uint16], min_consecutive: int = 3
) -> list[np.uint16]:
    """Locate the start indices of runs meeting the minimum consecutive length.

    Args:
        indices: Sorted array of integer indices.
        min_consecutive: Minimum number of consecutive values required to
            qualify as a run. Defaults to 3.

    Returns:
        List of onset indices where each qualifying run begins.

    Example:
        ```python
        idx = np.array([1, 2, 3, 5, 6, 10, 11, 12, 13], dtype=np.uint16)
        find_anomaly_onsets(idx, min_consecutive=3)
        ```

    """
    onsets = []
    # Group consecutive numbers
    for _, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
        group = list(g)
        if len(group) >= min_consecutive:
            onsets.append(group[0][1])

    return onsets


def convert_index_to_date(
    fouling_idx: list[np.uint16], residuals: pd.Series
) -> list[pd.Timestamp]:
    """Convert positional indices into their corresponding timestamps.

    Args:
        fouling_idx: List of integer positions into the residuals index.
        residuals: Series whose index is a DatetimeIndex of timestamps.

    Returns:
        List of timestamps located at each index position.

    Notes:
        AssertionError: If any resolved index value is not a pd.Timestamp.

    Example:
        ```python
        dates = convert_index_to_date([0, 5, 12], residuals)
        ```

    """
    fouling_dates = []
    for idx in fouling_idx:
        fouling_date = residuals.index[idx]
        assert isinstance(fouling_date, pd.Timestamp), (
            "fouling date must have pandas timestamp type!"
        )
        fouling_dates.append(fouling_date)
    return fouling_dates


def calculate_snr(
    residuals: pd.Series,
    fouling_date: pd.Timestamp,
    baseline_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"],
) -> tuple[float, float]:
    """Compute mean-based and peak-based signal-to-noise ratios for an event.

    Divides the magnitude of the fouling signal (mean or peak residual) by the
    baseline standard deviation to quantify how distinctly the fouling stands
    out from normal operation noise.

    Args:
      residuals: Residual signal indexed by datetime.
      fouling_date: Timestamp marking the onset of the fouling event.
      baseline_mask: Boolean mask identifying baseline (non-fouling) periods.
      anomaly_direction: Whether the fouling anomaly is positive ("pos") or
          negative ("neg") relative to the baseline.

    Returns:
      A tuple (snr, snr_peak) where snr uses the mean residual and snr_peak
      uses the maximum (or minimum) residual after the fouling date.

    Example:
        ```python
        snr, snr_peak = calculate_snr(
            residuals,
            fouling_date,
            baseline_mask,
            "neg",
        )
        ```

    """
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


def calculate_memoryless_cusum(
    residuals: pd.Series,
    shutin_mask: pd.Series,
    drift: float,
    target_mean: float,
    anomaly_direction: Literal["pos", "neg"],
) -> np.ndarray:
    """Calculate the Memoryless CUSUM statistic.

    Computes the CUSUM (Cumulative Sum) statistic to identify shifts in the mean
    of the residuals. It takes into account shut-in periods where the CUSUM is
    reset to zero and handles both positive and negative anomaly detection based
    on the specified direction.

    Args:
        residuals: Series of residual values.
        shutin_mask: Boolean mask indicating shut-in periods.
        drift: Drift value to subtract from target mean.
        target_mean: Target mean value for comparison.
        anomaly_direction: Direction of anomaly detection ("pos" or "neg").

    Returns:
        Array of CUSUM values.

    """
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


def predict_fouling_onset(
    residuals: pd.Series,
    baseline_mask: pd.Series,
    shutin_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"] = "neg",
    threshold_multiplier: float = 3,
    drift_multiplier: float = 0.5,
    target_mean: float = 0,
):
    """Predict the onset of fouling in a compressor.

    Uses residual data, baseline periods, and shut-in periods to detect
    anomalies that indicate potential fouling onset. It calculates the CUSUM
    statistic with resets during maintenance and determines if the cumulative
    sum of errors exceeds a threshold value, triggering an alarm. Additionally,
    it calculates the Signal-to-Noise Ratio (SNR) for detected anomalies.

    Args:
        residuals: Series of residual values.
        baseline_mask: Boolean mask indicating baseline periods.
        shutin_mask: Boolean mask indicating shut-in periods.
        anomaly_direction: Direction of anomaly detection ("pos" or "neg").
            Defaults to "neg".
        threshold_multiplier: Multiplier for determining the alarm threshold.
            Defaults to 3.
        drift_multiplier: Multiplier for determining the drift value. Defaults
            to 0.5.
        target_mean: Target mean value for comparison. Defaults to 0.

    Returns:
        A tuple (`fouling_dates`, `cusum`, `alarm_mask`) where `fouling_dates`,
        `cusum`, and `alarm_mask` are lists or arrays containing fouling dates,
        CUSUM values, and boolean masks indicating where the alarm threshold was
        exceeded.

    """
    # Baseline residuals are defined as the residuals during baseline period.
    # Baseline pressure drop is ~2.87 psi
    baseline_residuals: pd.Series = residuals.loc[baseline_mask]
    # CUSUM calculation with resets during maintenance
    # We ignore any residual deviation smaller than drift (0.5 * 2.87 = 1.44 psi)
    drift: float = drift_multiplier * baseline_residuals.std()
    # If the cumulative sum of errors exceeds threshold (3 * 2.87 = 8.61 psi) value,
    # trigger the alarm
    threshold: float = threshold_multiplier * baseline_residuals.std()
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
    is_no_alarm = not alarm_mask.any()
    if is_no_alarm:
        print("No fouling detected.")
        return None, cusum, alarm_mask
    fouling_idx = find_anomaly_onsets(np.where(alarm_mask)[0], min_consecutive=3)
    fouling_dates = convert_index_to_date(fouling_idx, residuals)
    for i, date in enumerate(fouling_dates):
        print(f"Fouling number {i + 1} detected: {date}")
        # Calculate SNR
        snr, snr_peak = calculate_snr(residuals, date, baseline_mask, anomaly_direction)
        print(f"Signal-to-Noise Ratio: {snr:.2f}")
        print(f"Peak SNR: {snr_peak:.2f}")
    return fouling_dates, cusum, alarm_mask


def plot_fouling_summary(
    y: pd.Series,
    y_pred: pd.Series,
    baseline_mask: pd.Series,
    shutin_mask: pd.Series,
    cusum: np.ndarray,
    alarm_mask: pd.Series,
    *,
    save: bool = False,
    fname: str | None = None,
    threshold_multiplier: float = 3.0,
):
    """Plot a fouling analysis summary.

    Generates a series of plots including actual vs baseline prediction,
    residuals from the baseline model, and CUSUM control chart to visualize
    anomalies in compressor fouling data. The plots highlight deviations,
    shut-in periods, thresholds for alarm conditions, and cumulative sum
    statistics to aid in detecting and analyzing fouling events.

    Args:
        y: Actual outlet pressure data.
        y_pred: Baseline model predictions.
        baseline_mask: Boolean mask for baseline period.
        shutin_mask: Boolean mask for shutdown periods.
        cusum: Cumulative sum of residuals.
        alarm_mask: Boolean mask for alarm periods.
        save: Whether to save the plot. Defaults to False.
        fname: File name if saving the plot. Defaults to None.
        threshold_multiplier: Multiplier for standard deviation to set
            thresholds. Defaults to 3.0.

    Raises:
        ValueError: If save is True and fname is None.

    """
    residuals = y - y_pred
    baseline_residuals = residuals[baseline_mask]
    threshold = threshold_multiplier * baseline_residuals.std()
    fig, axs = plt.subplots(3, 1, figsize=(25, 15), sharex=True)
    _ = fig.suptitle("Anomaly Detection Analysis", fontsize=16, fontweight="bold")

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
        label=f"±{threshold_multiplier}sigma Threshold",
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
        label=f"Decision Threshold: : {threshold_multiplier} sigma",
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

    plt.tight_layout()
    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    del fig, axs


def prepare_hierarchical_noise_args(
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Extract setpoint metadata for hierarchical noise modeling.

    Processes the 'Outlet_Pressure_SP' column in the input DataFrame to identify
    unique setpoints and creates mapping indices required for defining
    hierarchical noise structures (e.g., per-setpoint standard deviations).

    Args:
        X: Input data containing at least the 'Outlet_Pressure_SP' column.

    Returns:
        A tuple (setpoint_unique, setpoint_index, map_sp_to_idx) where
        setpoint_unique is a sorted array of unique setpoint values,
        setpoint_index is a NumPy array of an integer index mapping each row's
        setpoint to its position in 'setpoint_unique', and map_sp_to_idx is a
        dictionary mapping each setpoint value to its corresponding index in
        'setpoint_unique'.

    Example:
        ```python
        X = pd.DataFrame({"Outlet_Pressure_SP": [10.5, 12.3, 10.5]})
        setpoint_unique, setpoint_index, map_sp_to_idx = (
            extract_setpoint_metadata(X)
        )
        (array([10.5, 12.3]), array([0, 1, 0]), {10.5: 0, 12.3: 1})
        ```

    """
    setpoint_unique, setpoint_index = np.unique(
        X["Outlet_Pressure_SP"], return_inverse=True
    )
    map_sp_to_idx = {sp: i for i, sp in enumerate(setpoint_unique)}
    return setpoint_unique, setpoint_index, map_sp_to_idx


def calculate_empirical_sigma_stats(
    X: pd.DataFrame, y: pd.Series, setpoints: list[float]
):
    """Calculate empirical standard deviations and set up informed priors.

    Iterates over each setpoint, computes per-setpoint residual statistics
    (std, mean, count), and then aggregates across all setpoints to produce a
    weighted mean std, min/max std, and a range-to-mean ratio. The result
    informs the prior scale for Bayesian noise models.

    Args:
        X: Feature DataFrame that must contain `Outlet_Pressure_SP`.
        y: Target variable (residuals) aligned with X.
        setpoints: Unique setpoint values to iterate over.

    Returns:
        A tuple (empirical_stats, mean_std, range_to_mean_ratio, min_std,
        max_std), where empirical_stats maps each setpoint to a dict with
        keys ``std``, ``n``, and ``mean``, and the remaining four values are
        aggregate statistics across all setpoints.

    Example:
        ```python
        X = pd.DataFrame({"Outlet_Pressure_SP": [10.0, 20.0, 10.0]})
        y = pd.Series([0.1, 0.3, 0.15])
        stats, mean_std, ratio, min_std, max_std = (
            calculate_empirical_sigma_stats(X, y, [10, 20])
        )
        ```

    """
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
            f"Setpoint {sp:.1f} psi: sigma_unscaled = {y_subset.std():.3f} "
            + f"(n={len(y_subset)})"
        )

    # Calculate overall statistics
    all_stds: list[np.float64] = [stats["std"] for stats in empirical_stats.values()]
    weights: list[np.float64] = [stats["n"] for stats in empirical_stats.values()]
    mean_std: np.float64 = np.average(a=all_stds, weights=weights)
    min_std: np.float64 = np.min(all_stds)
    max_std: np.float64 = np.max(all_stds)
    range_to_mean_ratio: np.float64 = (max_std - min_std) / mean_std

    print("\n" + "=" * 60)
    print(
        f"Overall: weighted mean sigma_unscaled = {mean_std:.3f}, "
        + f"range normalized by mean = {100 * range_to_mean_ratio:.2f}% "
        + f"spread around mean = [{100 * (min_std - mean_std) / mean_std:.2f}%"
        + f", {100 * (max_std - mean_std) / mean_std:.2f}%]"
    )

    return empirical_stats, mean_std, range_to_mean_ratio, min_std, max_std


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
        X: Feature dataframe.
        y: Target series.
        shuffle_baseline: If True, shuffle data to break autocorrelation.
        use_offset: If True, compute setpoint offset.
        residual_target: If True, use residual from setpoint as target.
        random_seed: Random seef number.

    Returns:
        BayesianModelData with scaled features, target, and metadata.

    """
    X_shuffled: pd.DataFrame = X
    y_shuffled: pd.Series = y
    # Shuffle to break autocorrelation
    if shuffle_baseline:
        X_shuffled, y_shuffled = _shuffle_data(X, y, random_seed)
    setpoint_timeseries: pd.Series = X_shuffled.loc[:, "Outlet_Pressure_SP"]

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

    if residual_target:
        residual_obs: pd.Series = y_shuffled - setpoint_timeseries
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
    X: pd.DataFrame, y: pd.Series, random_seed: int | None
) -> tuple[pd.DataFrame, pd.Series]:
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
    features = X if not exclude_setpoint else X.drop(columns="Outlet_Pressure_SP")

    X_mean: pd.Series = features.mean(axis=0)
    X_std: pd.Series = features.std(axis=0)
    X_scaled: pd.DataFrame = (features - X_mean) / X_std

    return X_scaled, X_mean, X_std


def _scale_target(y: pd.Series) -> tuple[pd.Series, float, float]:
    y_mean = y.mean()
    y_std: float = cast(float, y.std())
    y_scaled = (y - y_mean) / y_std

    return y_scaled, y_mean, y_std


def _print_autocorr_change(y_original: pd.Series, y_scaled: pd.Series) -> None:
    print(
        f"autocorrelation changed from original {y_original.autocorr()} to "
        + f"{y_scaled.autocorr()}."
    )


def visualize_density_clusters(
    y_obs: Float64Matrix1D,
    bound_width: int | None,
    hist_bins: int | None,
    save: bool = False,
    fname: str | None = None,
):
    """Visualize the distribution of observed data and identify clusters.

    Computes a KDE, detects its peaks as candidate cluster centers, and also
    fits a Gaussian Mixture Model for comparison. The first subplot shows the
    histogram, KDE curve, and vertical markers for each detected center; the
    second subplot shows the empirical CDF.

    Args:
        y_obs: Array of observed (scaled) data values.
        bound_width: Bandwidth parameter for the KDE. Defaults to None.
        hist_bins: Number of bins for the histogram. Defaults to None.
        save: Whether to save the figure to disk. Defaults to False.
        fname: Path to save the figure if ``save`` is True. Defaults to
            None.

    Raises:
        ValueError: If save is True and fname is None.

    Example:
        ```python
        data = np.random.normal(0, 1, 500)
        visualize_density_clusters(data, bound_width=0.5, hist_bins=30)
        ```

    """
    kde = stats.gaussian_kde(y_obs, bw_method=bound_width)
    y_grid = np.linspace(y_obs.min() - 0.5, y_obs.max() + 0.5, 300)
    kde_densities = kde.evaluate(y_grid)
    peaks, _ = scipy.signal.find_peaks(kde_densities, distance=25)
    centers_kde = y_grid[peaks].round(3)
    # find center of each cluster (representing each setpoint)
    # GMM is more appropriate when clusters have different widths and densities
    gmm = GaussianMixture(n_components=4, random_state=14)
    _ = gmm.fit(y_obs.reshape(-1, 1))
    centers_gmm = (
        None if gmm.means_ is None else sorted(gmm.means_.round(3).flatten().tolist())
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
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
    if centers_gmm is None:
        print("Cluster means are not found.")
    else:
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
    plt.tight_layout()
    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, axs


class CoefPrior(Enum):
    """Enumeration of coefficient prior distributions for Bayesian regression.

    Attributes:
        NORMAL: Normal (Gaussian) prior distribution.
        LAPLACE: Laplace prior distribution for sparse solutions.

    """

    NORMAL = "normal"
    LAPLACE = "laplace"


class NoiseStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes:
        SINGLE: Single global noise parameter across all observations.
        HIERARCHICAL: Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class InterceptStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes:
        SINGLE: Single global noise parameter across all observations.
        HIERARCHICAL: Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class CoefficientStructure(Enum):
    """Enumeration of noise model types for Bayesian regression.

    Attributes:
        SINGLE: Single global noise parameter across all observations.
        HIERARCHICAL: Hierarchical noise model with per-group parameters.

    """

    NON_HIERARCHICAL = "non-hierarchical"
    HIERARCHICAL = "hierarchical"


class LikeLiHood(Enum):
    """Enumeration of likelihood model types for Bayesian regression.

    Attributes:
        NORMAL: Normal (Gaussian) distribution.
        T: t-student distribution.

    """

    NORMAL = "normal"
    T = "t-student"


def _build_coords(
    X: pd.DataFrame,
    has_setpoint_coords: bool = False,
    hierarchical_kwargs: HierarchicalModelParams | None = None,
):
    coords = {
        "obs": np.arange(len(X)),
        "predictors": X.columns,
    }
    if has_setpoint_coords:
        assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
        assert "setpoint_unique" in hierarchical_kwargs, (
            "hierarchical_kwargs must contain `setpoint_unique`"
        )
        n_sp = len(hierarchical_kwargs["setpoint_unique"])
        coords["setpoint"] = np.arange(n_sp)
    return coords


def _add_data(
    X: np.ndarray,
    y: Float64Matrix1D,
    has_setpoint_coords: bool = False,
    hierarchical_kwargs: HierarchicalModelParams | None = None,
    offset: pd.Series | None = None,
):
    X_data = cast(
        TensorSharedVariable, pm.Data(name="X", value=X, dims=("obs", "predictors"))
    )
    y_data = cast(TensorSharedVariable, pm.Data("y", y, dims="obs"))
    offset_data = (
        cast(TensorSharedVariable, pm.Data("offset", offset, dims="obs"))
        if offset is not None
        else None
    )
    sp_idx_data = None
    if has_setpoint_coords:
        assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
        assert "setpoint_index" in hierarchical_kwargs, (
            "hierarchical_kwargs must contain the `setpoint_index` key."
        )
        assert len(hierarchical_kwargs["setpoint_index"]) == y.size, (
            f"`setpoint_index` key must contain {y.size} indices."
        )
        sp_idx_data = cast(
            TensorSharedVariable,
            pm.Data(
                "sp_idx",
                hierarchical_kwargs["setpoint_index"],
                dims="obs",
            ),
        )

    return X_data, y_data, sp_idx_data, offset_data


def _add_noise(
    rng: np.random.Generator,
    noise_structure: NoiseStructure,
    noise_kwargs: dict[str, float],
    hierarchical_kwargs: HierarchicalModelParams | None = None,
    sp_idx_data: TensorSharedVariable | None = None,
):
    # sigma (represeneted here with τ to avoid confusion) controls
    # multiplicative uncertainty: log sigma ~ N(log(sigma_{0}), τ) => sigma ~
    # LogNormal(log(sigma_{0}), τ) => sigma=sigma_{0}⋅exp(ϵ), ϵ ~ N(0,τ) Suppose
    # we believe "sigma is likely within ±X% of sigma₀" => τ = log(1 + X)
    if noise_structure == NoiseStructure.NON_HIERARCHICAL:
        # Aleatoric uncertainty (measurement noise)
        log_sigma = pm.Normal(
            "log_sigma",
            mu=np.log(noise_kwargs["sigma_mu_mu"]),
            sigma=noise_kwargs["sigma_mu_sd"],
            rng=rng,
        )
        # Scalar, used directly in likelihood
        sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
        return pt.as_tensor(sigma)
    # hierarchical uncertanity
    assert hierarchical_kwargs is not None, "hierarchical_kwargs is not given!"
    assert sp_idx_data is not None, "sp_idx_data required for hierarchical!"
    # group-level mean & sd
    log_sigma_mu = pm.Normal(
        "log_sigma_mu",
        mu=np.log(noise_kwargs["sigma_mu_mu"]),
        sigma=noise_kwargs["sigma_mu_sd"],
        rng=rng,
    )
    # Let the sigma for each setpoint vary around the group mean, with expected
    # multiplicative differences of up double the sigma spread (i.e., 12%),
    # unless the data indicate otherwise.
    log_sigma_sd = pm.HalfNormal(
        "log_sigma_sd", sigma=hierarchical_kwargs["sigma_sd_sd"], rng=rng
    )
    # non-centered per-setpoint raw
    log_sigma_raw = pm.Normal("log_sigma_raw", 0, 1, dims="setpoint", rng=rng)
    # per-setpoint log-sigma
    log_sigma_sp = log_sigma_mu + log_sigma_raw * log_sigma_sd
    # log_sigma_sp = pm.Deterministic(
    #     "log_sigma_sp",
    #     log_sigma_mu + log_sigma_raw * log_sigma_sd,
    #     dims="setpoint",
    # )
    # unique sigmas (stored in inference data)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma_sp), dims="setpoint")
    # Return indexed version for likelihood (not stored separately)
    return pt.as_tensor(sigma)[sp_idx_data]


def _add_intercept(
    intercept_sd: float,
    intercept_structure: InterceptStructure,
    sp_idx_data: TensorSharedVariable | None = None,
):
    if intercept_structure is InterceptStructure.HIERARCHICAL:
        assert sp_idx_data is not None, "sp_idx_data required for hierarchical!"
        mu_intercept = pm.Normal("mu_intercept", 0.0, intercept_sd)
        sigma_intercept = pm.HalfNormal("sigma_intercept", 0.5 * intercept_sd)
        intercept = pm.Normal(
            "intercept", mu=mu_intercept, sigma=sigma_intercept, dims="setpoint"
        )
        return cast(TensorVariable[Any, Any], intercept[sp_idx_data])  # pyright: ignore[reportExplicitAny]
    return pm.Normal("intercept", mu=0.0, sigma=intercept_sd)


def _add_coefficient_priors(
    coef_prior: Literal[CoefPrior.NORMAL, CoefPrior.LAPLACE],
    coef_kwargs: dict[str, float],
    coefficient_structure: CoefficientStructure,
):
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
        return pt.as_tensor(beta), True

    # Shared betas
    if coef_prior is CoefPrior.NORMAL:
        return pm.Normal(
            "beta",
            mu=coef_kwargs.get("mu", 0.0),
            sigma=coef_kwargs.get("sd", 0.5),
            dims="predictors",
        ), False
    # LAPLACE
    # Laplace variance = 2b²
    tau = pm.HalfNormal("tau", sigma=0.5 * np.sqrt(coef_kwargs.get("sd", 0.5)))
    return pm.Laplace(
        "beta",
        mu=coef_kwargs.get("mu", 0.0),
        b=tau,
        dims="predictors",
    ), False


def _add_likelihood(
    likelihood_model: Literal[LikeLiHood.NORMAL, LikeLiHood.T],
    intercept: TensorVariable[Any, Any],  # pyright: ignore[reportExplicitAny]
    beta: TensorVariable[Any, Any],  # pyright: ignore[reportExplicitAny]
    sigma: TensorVariable[Any, Any],  # pyright: ignore[reportExplicitAny]
    X_data: TensorSharedVariable,
    y_data: TensorSharedVariable,
    *,
    like_var: str = "y_like",
    sp_idx_data: TensorSharedVariable | None = None,
    likelihood_model_kwargs: dict[str, float] | None = None,
    offset_data: TensorSharedVariable | None = None,
    beta_has_setpoint_dim: bool = False,
    rng: np.random.Generator | None = None,
):
    # Compute linear predictor
    if beta_has_setpoint_dim and sp_idx_data is not None:
        beta_obs = beta[:, sp_idx_data]  # (n_predictors, setpoint)
        mu_lin = (X_data * beta_obs.T).sum(axis=1)  # (n_predictors,)
    else:
        mu_lin = pt.dot(X_data, beta)  # (n_obs,)

    # Compute mean with intercept
    base = offset_data if offset_data is not None else 0.0
    mu = pm.Deterministic("mu", base + intercept + mu_lin, dims="obs")

    # Create likelihood
    if likelihood_model == LikeLiHood.NORMAL:
        return pm.Normal(
            like_var, mu=mu, sigma=sigma, observed=y_data, rng=rng, dims="obs"
        )
    if likelihood_model_kwargs is None:
        raise ValueError("likelihood_model_kwargs required for Student-T likelihood")
    nu = pm.Gamma(
        "nu",
        alpha=likelihood_model_kwargs.get("alpha", 2.0),
        beta=likelihood_model_kwargs.get("beta", 0.1),
        rng=rng,
    )
    return pm.StudentT(
        like_var, nu=nu, mu=mu, sigma=sigma, observed=y_data, rng=rng, dims="obs"
    )


def _sample_and_build_idata(
    model: pm.Model,
    draws: int,
    tune: int,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] | None,
    target_accept: float,
    rng: np.random.Generator,
    initvals: StartDict | Sequence[StartDict | None] | None = None,
):
    # Sample and assemble InferenceData
    with model:
        # draw the posterior
        idata = pm.sample(
            draws=draws,
            tune=tune,
            nuts_sampler=nuts_sampler,
            initvals=initvals,
            target_accept=target_accept,
            progressbar=False,
            quiet=True,
            random_seed=rng,
        )
        # prior predictive
        prior = pm.sample_prior_predictive(draws=draws, random_seed=rng)
        # DataTree uses update() not extend()
        idata.update(prior)  # type: ignore[basedpyright reportAttributeAccessIssue]
        # posterior predictive
        ppc = pm.sample_posterior_predictive(idata, progressbar=False, random_seed=rng)
        idata.update(ppc)  # type: ignore[basedpyright reportAttributeAccessIssue]
        # log likelihood
        _ = pm.compute_log_likelihood(idata)
    return idata


def build_bayesian_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    like_var: str = "y_like",
    has_setpoint_coords: bool = False,
    coef_prior: Literal[CoefPrior.NORMAL, CoefPrior.LAPLACE] = CoefPrior.NORMAL,
    noise_structure: Literal[
        NoiseStructure.HIERARCHICAL, NoiseStructure.NON_HIERARCHICAL
    ] = NoiseStructure.NON_HIERARCHICAL,
    intercept_structure: Literal[
        InterceptStructure.HIERARCHICAL, InterceptStructure.NON_HIERARCHICAL
    ] = InterceptStructure.NON_HIERARCHICAL,
    coefficient_structure: Literal[
        CoefficientStructure.HIERARCHICAL, CoefficientStructure.NON_HIERARCHICAL
    ] = CoefficientStructure.NON_HIERARCHICAL,
    likelihood_model: Literal[LikeLiHood.NORMAL, LikeLiHood.T] = LikeLiHood.NORMAL,
    intercept_sd: float = 1.0,
    coef_kwargs: dict[str, float] | None = None,
    noise_kwargs: dict[str, float] | None = None,
    hierarchical_kwargs: HierarchicalModelParams | None = None,
    likelihood_model_kwargs: dict[str, float] | None = None,
    offset: pd.Series | None = None,
    random_seed: int | None = None,
) -> pm.Model:
    """Construct a Bayesian model using PyMC3.

    Args:
        X: DataFrame containing the features.
        y: Series containing the target variable.
        like_var: Name of the likelihood variable. Defaults to "y_like".

        has_setpoint_coords: Indicates if setpoint coordinates are present. Defaults to
            False.
        coef_prior: Prior distribution for coefficients. Defaults to CoefPrior.NORMAL.
        noise_structure: Structure of the noise model. Defaults to
            NoiseStructure.NON_HIERARCHICAL.
        intercept_structure: Structure of the intercept. Defaults to
            InterceptStructure.NON_HIERARCHICAL.
        coefficient_structure: Structure of the coefficients. Defaults to
        CoefficientStructure.NON_HIERARCHICAL.
        likelihood_model: Likelihood model type. Defaults to LikeLiHood.NORMAL.
        intercept_sd: Standard deviation for the intercept. Defaults to 1.0.
        coef_kwargs: Additional keyword arguments for coefficient priors.
        noise_kwargs: Additional keyword arguments for noise model.
        hierarchical_kwargs: Additional keyword arguments for hierarchical structures.
        likelihood_model_kwargs: Additional keyword arguments for likelihood model.
        offset: Offset value if any.
        random_seed: Seed for random number generator.

    Returns:
        The constructed PyMC3 Bayesian model.

    Example:
        ```python
         X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
         y = pd.Series([5, 6])
         build_bayesian_model(X, y)
         ```

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    coef_kwargs = coef_kwargs if coef_kwargs is not None else {}
    noise_kwargs = noise_kwargs if noise_kwargs is not None else {}

    # Build coords once, up front
    coords = _build_coords(X, has_setpoint_coords, hierarchical_kwargs)

    with pm.Model(coords=coords) as model:
        # Data containers
        X_data, y_data, sp_idx_data, offset_data = _add_data(
            X.to_numpy(), y.to_numpy(), has_setpoint_coords, hierarchical_kwargs, offset
        )
        # Intercept and coefficient priors
        intercept = _add_intercept(intercept_sd, intercept_structure, sp_idx_data)
        beta, beta_has_setpoint_dim = _add_coefficient_priors(
            coef_prior, coef_kwargs, coefficient_structure
        )
        # Noise model (single vs hierarchical)
        sigma = _add_noise(
            rng, noise_structure, noise_kwargs, hierarchical_kwargs, sp_idx_data
        )
        # Likelihood
        _ = _add_likelihood(
            likelihood_model,
            intercept,
            beta,
            sigma,
            X_data,
            y_data,
            like_var=like_var,
            sp_idx_data=sp_idx_data,
            likelihood_model_kwargs=likelihood_model_kwargs,
            offset_data=offset_data,
            beta_has_setpoint_dim=beta_has_setpoint_dim,
            rng=rng,
        )

    return model


def fit_bayesian_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "nutpie",
    initvals: StartDict | Sequence[StartDict | None] | None = None,
    target_accept: float = 0.9,
    random_seed: int | None = None,
) -> DataTree:
    """Fits a Bayesian model using the specified sampler.

    Args:
        model: The Bayesian model to fit.
        draws: Number of posterior samples to draw. Defaults to 2000.
        tune: Number of tuning steps for the NUTS sampler. Defaults to 1000.
        nuts_sampler: The sampler to use for sampling from the posterior. Defaults to
            'nutpie'.
        initvals : optional, dict, array of dict Dict or list of dicts with
            initial value strategies to use instead of the defaults from
            `Model.initial_values`. The keys should be names of transformed
            random variables.
        target_accept: Target acceptance probability for the NUTS sampler. Defaults to
            0.9.
        random_seed: Seed for the random number generator. If None, a random seed is
            used. Defaults to None.

    Returns:
        A xarray.Dataset containing the posterior samples.

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )

    return _sample_and_build_idata(
        model, draws, tune, nuts_sampler, target_accept, rng, initvals
    )


def build_and_fit_bayesian_models(
    model_configs: list[dict[str, Any]],  # pyright: ignore[reportExplicitAny]
    common_params: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> tuple[dict[str, pm.Model], dict[str, DataTree]]:
    """Build and fit bayesian models from a list of model configs.

    Each config in ``model_configs`` is unpacked as keyword arguments to
    :func:`build_bayesian_model`, with missing optional fields falling back
    to their default enum values. After construction, every model is fitted
    via :func:`fit_bayesian_model` using the ``nutpie`` sampler and the
    common random seed.

    Args:
      model_configs: Sequence of dicts, each containing at minimum ``name``,
        ``X``, and ``y``. Optional fields include ``like_var``,
        ``has_setpoint_coords``, ``coef_prior``, ``noise_structure``,
        ``intercept_structure``, ``coefficient_structure``,
        ``likelihood_model``, and ``offset``. Extra keys are forwarded via
        ``**common_params``.
      common_params: Shared keyword arguments passed to every
        :func:`build_bayesian_model` call. Must include ``random_seed``.

    Returns:
      A tuple ``(models, idatas)``, where ``models`` maps model names to
      PyMC model objects, and ``idatas`` maps model names to their
      corresponding ArviZ ``DataTree`` inference data objects.

    Example:
      ```python
      configs = [
          {"name": "m1", "X": X_train, "y": y_train},
          {
              "name": "m2",
              "X": X_train,
              "y": y_train,
              "noise_structure": NoiseStructure.HIERARCHICAL,
          },
      ]
      params = {"random_seed": 42}
      models, idatas = build_and_fit_bayesian_models(configs, params)
      ```

    """
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
    return models, idatas


def compare_and_select_best_model(
    models_dict: dict[str, pm.Model], idatas_dict: dict[str, DataTree]
) -> tuple[pm.Model, DataTree]:
    """Compare Bayesian models and return the best one by ELPD.

    Runs :func:`arviz_stats.loo.compare` with the
    ``BB-pseudo-BMA`` method on all inference data objects, displays the
    full comparison table, and returns the top-ranked model along with its
    corresponding ``DataTree``.

    Args:
      models_dict: Dictionary mapping model names to PyMC model objects.
      idatas_dict: Dictionary mapping model names to ArviZ ``DataTree``
        inference data objects, keyed identically to ``models_dict``.

    Returns:
      A tuple ``(model_best, idata_best)``, where ``model_best`` is the
      PyMC model with the highest ELPD rank, and ``idata_best`` is its
      corresponding inference data object.

    Example:
      ```python
      model_best, idata_best = compare_and_select_best_model(
          models,
          idatas,
      )
      ```

    """
    compare_results = compare(idatas_dict, method="BB-pseudo-BMA", round_to=2)  # pyright: ignore[reportArgumentType]
    _ = display(compare_results)

    # Select best model and its idata
    best_model_name = cast(str, compare_results.index[0])  # Top-ranked model
    model_best = models_dict[best_model_name]
    idata_best = idatas_dict[best_model_name]

    return model_best, idata_best


def _marginal_density_single_draw(
    mu_vec: Array, sigma_scalar: ScalarLike, y_g: Array
) -> Array:
    """Compute marginal density for one posterior draw.

    Computes the marginal probability density over a grid `y_g` for a single
    posterior sample. The marginal density is obtained by averaging the normal
    density of each observation across all observations.

    Args:
        mu_vec: shape (N,), containing posterior means for each observation in
            the draw.
        sigma_scalar: scalar, the posterior standard deviation for the draw
            (shared by all observations in this draw).
        y_g: shape (G,), the grid of observation values (the same grid `y` is
            used for every draw).


    Returns:
            shape (G,), marginal density evaluated at each grid point.

    Example:
            ```python
            import jax.numpy as jnp
            from utility import _marginal_density_single_draw

            mu = jnp.array([0.0, 1.0])
            sigma = jnp.array(0.5)
            y = jnp.linspace(-2, 2, 50)
            dens = _marginal_density_single_draw(mu, sigma, y)
            print(dens.shape)  # (50,)
            ```

    """
    per_obs = jax_norm.pdf(
        y_g[:, None],  # (G, 1)
        loc=mu_vec[None, :],  # (1, N)
        scale=sigma_scalar,
    )  # (G, N) — density at each grid point for each observation
    # Average over observations → marginal density per sample
    # how likely each grid point is when we *marginalize* (i.e. average) out the
    # individual observation-specific densities.
    return per_obs.mean(axis=1)  # (G,)


def _plot_interactive_posterior_draw(
    mu_post: np.ndarray,
    sigma_post: np.ndarray,
    y_grid: np.ndarray,
    rng: np.random.Generator,
) -> go.Figure:
    """Plot per-observation densities for a single posterior draw.

    Plots each observation's normal density curve for a single randomly
    selected posterior sample, along with the mean density across all
    observations. The figure includes show-all and hide-all toggle buttons
    and a title displaying the sample index and theoretical maximum density.

    Args:
      mu_post: Posterior means with shape ``(n_observations, n_samples)``.
      sigma_post: Posterior standard deviations with shape ``(n_samples,)``.
      y_grid: Dense 1-D grid of y-values over which densities are evaluated.
      rng: NumPy random generator used to select the posterior draw to plot.

    Returns:
      A Plotly ``Figure`` with one trace per observation and a mean curve.

    """
    n_observations = mu_post.shape[0]
    n_posterior_samples = sigma_post.size
    s = rng.choice(n_posterior_samples)
    fmax = 1 / (sigma_post[s] * np.sqrt(2 * np.pi))

    fig = go.Figure()

    colors = [
        f"hsl({int(h)}, 70%, 50%)"
        for h in np.linspace(0, 360 * (129 / n_observations), n_observations)
    ]

    for obs_idx in range(n_observations):
        d = stats.norm.pdf(y_grid, mu_post[obs_idx, s], sigma_post[s])
        _ = fig.add_trace(
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

    mean_d = np.mean(
        np.array(
            [
                stats.norm.pdf(y_grid, mu_post[obs_idx, s], sigma_post[s])
                for obs_idx in range(n_observations)
            ]
        ),
        axis=0,
    )

    _ = fig.add_trace(
        go.Scatter(
            x=y_grid,
            y=mean_d,
            mode="lines",
            name="Mean",
            line={"color": "black", "width": 3, "dash": "dash"},
            opacity=1.0,
            hovertemplate="Mean: %{y:.3f}<extra></extra>",
        )
    )
    _ = fig.update_layout(
        title={
            "text": (
                f"Posterior Predictive Distributions — Sample {s}"
                f"<br><sup>f<sub>max</sub> = 1 / (sigma.√2π) = 1 / "
                f"({sigma_post[s]:.3f} . √2π) = {fmax:.3f}</sup>"
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
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
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
            "range": [0, np.ceil(fmax * 1.10)],
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

    return fig


def _render_pc_to_img(
    pc: PlotCollection,
) -> np.ndarray:
    """Render an arviz_plots ``PlotCollection`` figure to a numpy image array.

    Extracts the matplotlib figure from the ``PlotCollection``, saves it
    to a bytes buffer, and reads it back as an RGBA array for embedding
    into another figure.

    Args:
      pc: ArviZ ``PlotCollection`` containing a figure in
        ``pc.viz.ds["figure"]``.

    Returns:
      The rendered figure as an ``(H, W, 4)`` RGBA numpy array.

    Example:
      ```python
      img = _render_pc_to_img(pc)
      ```

    """
    ecdf_fig = pc.viz.ds["figure"].to_numpy().item()
    buf = io.BytesIO()
    ecdf_fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    _ = buf.seek(0)
    ecdf_img = plt.imread(buf)
    plt.close(ecdf_fig)
    return ecdf_img


def _compute_predictive_and_marginal_densities(
    idata: DataTree,
    n_samples_to_plot: int,
    var_name: str,
    rng: np.random.Generator,
):
    """Draw posterior predictive samples and compute marginal densities.

    Extracts posterior ``mu`` and ``sigma`` from ``idata``, draws
    predictive samples from the normal observation model, and computes
    per-draw marginal densities by averaging per-observation normal PDFs
    over a shared evaluation grid. Both the pooled predictive histogram
    and the individual marginal density curves are used in
    ``plot_posterior_predictive``.

    Args:
      idata: ArviZ ``DataTree`` containing a ``posterior`` group with
        ``mu`` and ``sigma``, and an ``observed_data`` group with the
        variable given by ``var_name``.
      n_samples_to_plot: Number of posterior draws to subsample for
        marginal density computation.
      var_name: Name of the observed variable in
        ``idata.observed_data``.
      rng: NumPy random generator for reproducible sampling.

    Returns:
      A tuple ``(mu_post, sigma_post, y_true, y_pred, y_grid,
      marginal_densities)``, where ``mu_post`` is ``(N, n_draws)``,
      ``sigma_post`` is ``(n_draws,)``, ``y_true`` is the observed data
      ``(N,)``, ``y_pred`` is the pooled posterior predictive sample
      ``(N, n_draws)``, ``y_grid`` is the evaluation grid ``(G,)``, and
      ``marginal_densities`` is ``(G, S)`` with ``S =
      n_samples_to_plot``.

    Example:
      ```python
      mu_post, sigma_post, y_true, y_pred, y_grid, marginals = (
          _compute_predictive_and_marginal_densities(
              idata, n_samples_to_plot=500, var_name="y_like", rng=rng
          )
      )
      ```

    """
    posterior_samples_stacked = extract(
        idata, group="posterior", var_names=["mu", "sigma"], combined=True
    )
    mu_post: Float64Matrix2D = (
        posterior_samples_stacked.mu.to_numpy()
    )  # (N, chains * draws)
    sigma_post: Float64Matrix1D = (
        posterior_samples_stacked.sigma.to_numpy()
    )  # (chains * draws,)
    y_true: Float64Matrix1D = idata.observed_data[var_name].to_numpy()

    y_pred = rng.normal(loc=mu_post, scale=sigma_post)
    n_posterior_samples = sigma_post.size
    y_grid = cast(
        Float64Matrix1D, np.linspace(y_true.min() - 0.5, y_true.max() + 0.5, 300)
    )
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
    #     # 1/n sum_{i=1}_{i=n} Normal(y | μ_i_{^s}, sigma_{^s}) for this draw s
    #     marginal_density = np.mean(
    #         [
    #             stats.norm.pdf(y_grid, mu_post[i, s], sigma_post[s])
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
    return mu_post, sigma_post, y_true, y_pred, y_grid, marginal_densities


def plot_posterior_predictive(
    idata: DataTree,
    n_samples_to_plot: int = 1000,
    var_name: str = "y_like",
    figsize: tuple[float, float] = (20, 5),
    random_seed: int | None = None,
):
    """Plot posterior predictive distributions against observed data.

    Draws samples from the posterior predictive distribution using the
    posterior ``mu`` and ``sigma``, compares them to the observed data via
    a static histogram overlay, and visualises the marginal density of
    each posterior draw on a single matplotlib figure. A second interactive
    Plotly figure breaks down the per-observation normal densities for one
    sampled posterior draw, with toggleable traces and a mean curve.

    The static ``fig`` contains two side-by-side subplots:

    * **ECDF (left)**: An empirical cumulative distribution function plot
      generated by ``arviz_plots.plot_ppc_dist``, comparing the observed
      data ECDF against posterior predictive ECDFs.
    * **KDE (right)**: A density overlay with three elements — a histogram
      of the observed data (white), a histogram of all pooled posterior
      predictive samples (red), and per-draw marginal densities computed
      by averaging per-observation normals across the dataset (blue
      lines, one per posterior draw).

    Args:
      idata: ArviZ ``DataTree`` containing ``posterior`` groups with
        ``mu`` and ``sigma``, and an ``observed_data`` group with the
        variable named by ``var_name``.
      n_samples_to_plot: Number of posterior draws to overlay on the
        matplotlib histogram. Defaults to 1000.
      var_name: Name of the observed variable in ``idata.observed_data``.
        Defaults to ``"y_like"``.
      figsize: Matplotlib figure dimensions ``(width, height)`` in inches.
        Defaults to ``(20, 5)``.
      random_seed: Seed for reproducible random sampling. Defaults to
        None.

    Returns:
      A tuple ``(fig, fig_interactive)``, where ``fig`` is the static
      matplotlib figure with the histogram and marginal density overlay,
      and ``fig_interactive`` is the Plotly figure with per-observation
      normal densities for a single posterior draw.

    Example:
      ```python
      fig, fig_interactive = plot_posterior_predictive(
          idata,
          n_samples_to_plot=500,
          random_seed=42,
      )
      ```

    """
    rng = np.random.default_rng(random_seed)

    mu_post, sigma_post, y_true, y_pred, y_grid, marginal_densities = (
        _compute_predictive_and_marginal_densities(
            idata, n_samples_to_plot, var_name, rng
        )
    )
    pc = azp.plot_ppc_dist(
        idata,
        kind="ecdf",
        num_samples=n_samples_to_plot,
        figure_kwargs={"figsize": (10, 5)},
        visuals={"title": False},
    )
    ecdf_img = _render_pc_to_img(pc)

    fig, _ = plt.subplots(1, 2, figsize=figsize)
    _ = fig.suptitle("Posterior Predictive Check")
    # ecdf plot
    _ = fig.axes[0].imshow(ecdf_img)
    _ = fig.axes[0].set_title("ecdf")
    _ = fig.axes[0].axis("off")
    # kde plot: 1. observation distribution
    _ = fig.axes[1].hist(y_true, bins=50, density=True, alpha=0.5, color="white")
    # kde plot: 2. pooled posterior predictive
    _ = fig.axes[1].hist(
        y_pred.flatten(), bins=50, density=True, alpha=0.3, color="red"
    )
    # kde plot: 3. marginal densities
    _ = fig.axes[1].plot(
        y_grid,
        marginal_densities,
        color="cyan",
        alpha=0.05,
    )
    _ = fig.axes[1].set_xlabel("Scaled outlet pressure")
    _ = fig.axes[1].set_ylabel("Density")
    _ = fig.axes[1].set_title("kde")
    _ = fig.axes[1].legend(
        [
            Line2D([0], [0], color="white", lw=4),
            Line2D([0], [0], color="red", lw=4),
            Line2D([0], [0], color="cyan", lw=4),
        ],
        [
            "Observed data",
            "Posterior predictive (pooled)",
            "Marginal densities",
        ],
    )
    plt.show()

    fig_interactive = _plot_interactive_posterior_draw(mu_post, sigma_post, y_grid, rng)
    fig_interactive.show()

    return fig, fig_interactive


def check_likelihood_qqplot(
    idata: DataTree,
    likelihood: Literal["normal", "t-student"],
    y_true: Float64Matrix1D,
    dim: tuple[str, str] = ("chain", "draw"),
) -> Figure:
    """Check the observation-model fit via a Q-Q plot of standardized residuals.

    Standardises residuals as ``(y_true - mu) / sigma`` using the posterior
    mean of ``mu`` and ``sigma``, then draws a Q-Q plot against the normal
    distribution. For a Student-t likelihood the degrees-of-freedom
    parameter ``nu`` is passed to ``scipy.stats.probplot`` so that the
    theoretical quantiles account for heavier tails.

    Args:
      idata: ArviZ ``DataTree`` containing a ``posterior`` group with
        ``mu`` and ``sigma`` (and ``nu`` if ``likelihood == "t-student"``).
      likelihood: Observation model to validate. Determines whether ``nu``
        is extracted from the posterior.
      y_true: Observed response variable.
      dim: Sample dimensions to average over when computing posterior
        means. Defaults to ``("chain", "draw")``.

    Returns:
      A matplotlib ``Figure`` with the Q-Q plot.

    Example:
      ```python
      fig = check_likelihood_qqplot(
          idata, likelihood="normal", y_true=y, dim=("chain", "draw")
      )
      fig.savefig("qq.png")
      ```

    """
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
    sparams: tuple[float, ...] = ()
    if likelihood == "t-student":
        nu_mean = mcmc_samples.nu.mean(dim=dim)  # scaler
        sparams = (float(nu_mean),)
    # standardize residuals (should be ~Normal(0, 1))
    residuals = (y_true - mu_mean) / sigma_mean
    fig, ax = plt.subplots(1, 1)
    stats.probplot(  # pyright: ignore[reportCallIssue]
        residuals.to_numpy(),
        dist="norm",
        sparams=sparams,
        plot=ax,  # pyright: ignore[reportArgumentType]
        fit=True,
        rvalue=True,
    )
    _ = ax.set_title(
        f"Q-Q Plot of Residuals Given {likelihood} likelihood", fontsize=12
    )
    return fig


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

    mu_raw: DataArray  # (chain, draw, obs)
    mu_stacked: DataArray  # (obs, n_samples)
    sigma_raw: DataArray  # (chain, draw)
    sigma_stacked: DataArray  # (n_samples,)
    y_pred_raw: DataArray  # (chain, draw, obs)
    y_pred_stacked: DataArray  # (obs, n_samples)
    # y_obs: Float64Matrix1D  # (obs, )
    y_obs: DataArray
    log_likelihood_stacked: Float64Matrix2D  # (n_obs, n_samples)


def _stack_samples(da: xr.DataArray):
    """Stack chain and draw dimensions into a single sample dimension.

    Only stacks dimensions that actually exist in the DataArray, making it
    safe to call on arrays that may already be combined.

    Args:
      da: DataArray with optional ``chain`` and/or ``draw`` dimensions.

    Returns:
      DataArray with a ``sample`` dimension replacing the stacked dims.

    Raises:
      ValueError: If neither ``chain`` nor ``draw`` is present in
        ``da.dims``.

    """
    dims_to_stack = [d for d in ("chain", "draw") if d in da.dims]
    if not dims_to_stack:
        raise ValueError(f"No chain/draw dims found. Got: {da.dims}")
    return da.stack(sample=tuple(dims_to_stack))  # noqa: PD013 - stack() required for xarray dimension stacking


def exctract_pymc_groups_data(idata: DataTree) -> MCMCSamples:
    """Extract posterior, predictive, and log-likelihood from a PyMC DataTree.

    Pulls ``mu``, ``sigma``, posterior predictive samples, and log
    likelihood from ``idata``, returning them in both raw (chain/draw)
    and stacked (sample) formats for convenient downstream use.

    Args:
      idata: ArviZ ``DataTree`` containing ``posterior``,
        ``posterior_predictive``, ``observed_data``, and
        ``log_likelihood`` groups.

    Returns:
      An ``MCMCSamples`` named-tuple-like container with ``mu_raw``,
      ``mu_stacked``, ``sigma_raw``, ``sigma_stacked``, ``y_pred_raw``,
      ``y_pred_stacked``, ``y_obs``, and ``log_likelihood_stacked``.

    Example:
      ```python
      samples = exctract_pymc_groups_data(idata)
      mu = samples.mu_stacked
      sigma = samples.sigma_stacked
      ```

    """
    # Extract mu and sigma
    mcmc_samples = extract(
        idata,
        group="posterior",
        var_names=["sigma", "mu"],
        combined=False,
    )
    mu_raw = mcmc_samples.mu
    sigma_raw = mcmc_samples.sigma
    mu_stacked = _stack_samples(mcmc_samples.mu)
    sigma_stacked = _stack_samples(mcmc_samples.sigma)
    # Exctract observed values
    obs_dim = next(iter(idata.observed_data.dims.keys()))
    y_obs = extract(idata, group="observed_data", sample_dims=obs_dim)
    y_obs = cast(DataArray, y_obs)
    # Extract y predictions
    y_pred_raw = extract(idata, group="posterior_predictive", combined=False)
    y_pred_raw = cast(DataArray, y_pred_raw)
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


def compute_psis_weights(
    ll: Float64Matrix2D,
) -> tuple[Float64Matrix2D, Float64Matrix1D]:
    """Compute PSIS weights to approximate LOO posterior.

    Computes PSIS weights using the input log-likelihood matrix.
    It first calculates the Pareto k values and then normalizes the weights for
    each observation considering numerical stability. The function returns the
    normalized weights and the Pareto k values.

    Args:
        ll: Float64Matrix2D
            Input log-likelihood matrix with shape (n_obs, n_samples).

    Returns:
        tuple[Float64Matrix2D, Float64Matrix1D]
            A tuple containing:
            - weights: Normalized PSIS weights with shape (n_obs, n_samples).
            - pareto_k: Pareto k values with shape (n_obs, ).

    Warns:
        UserWarning
            If any Pareto k value exceeds 0.7, indicating potential issues
            with LOO estimate reliability. A hard assertion is avoided because
            Pareto-k estimates are sensitive to MCMC sampling variability —
            a single run can cross the 0.7 threshold randomly. Callers should
            inspect the returned ``pareto_k`` array for their use case.

    """
    # Compute PSIS weights (these reweight posterior samples to approximate LOO
    # posterior)
    result = array_stats.psislw(-ll)
    assert result is not None, "psis method has returned None."
    log_weights, pareto_k = cast(tuple[Float64Matrix2D, Float64Matrix1D], result)

    pareto_k_max = cast(float, np.max(pareto_k))
    if pareto_k_max > 0.7:
        warnings.warn(
            "PSIS Pareto k values indicate potential issues with LOO estimates: "
            + f"max k = {pareto_k_max:.3f} (threshold = 0.7). "
            + f"{(pareto_k > 0.7).sum()} of {len(pareto_k)} "
            + "observations exceed the threshold.",
            stacklevel=2,
        )
    # log_weights shape (n_obs, n_samples) - log weights for each observation
    # pareto_k shape (n_obs, ) - pareto_k for each observation
    # Normalize weights for each observation considering numerical stability
    weights: Float64Matrix2D = np.exp(
        log_weights - cast(Float64Matrix2D, np.max(log_weights, axis=1, keepdims=True))
    )
    weights /= np.sum(weights, axis=1, keepdims=True)  # (n_obs, n_samples)

    return weights, pareto_k


@dataclass
class ELPDMetrics:
    """Container for ELPD evaluation metrics and diagnostic values.

    Attributes:
      pointwise_densities: Pointwise ELPD contributions ``exp(elpd_i)``.
      avg_loo_log_density: Average out-of-sample log-density ``ELPD / n``.
      geom_mean_density: Geometric mean density ``exp(avg_loo_log_density)``.
      analytical_log_density_dist: Ideal log-densities across posterior
        draws of ``sigma``.
      ideal_mean_density: Expected density under perfect calibration.
      deviation: Root-mean-square standardized residual between expected
        and actual log-density.
      trouble_obs_indices: Indices of poorly predicted observations.
      well_predicted_indices: Indices of the best-predicted observations.
      obs_to_plot: Concatenation of trouble and well-predicted indices.

    """

    pointwise_densities: Float64Matrix1D  # (n_obs, )
    avg_loo_log_density: float
    geom_mean_density: float
    analytical_log_density_dist: Float64Matrix1D  # (n_samples [chain * draw],)
    ideal_mean_density: np.float64
    deviation: np.float64
    trouble_obs_indices: Uint16Matrix1D
    well_predicted_indices: Uint16Matrix1D
    obs_to_plot: Uint16Matrix1D


def _compute_elpd_metrics(
    elpd_data: ELPDData,
    posterior_sigma: Float64Matrix1D,
    trouble_obs_indices: Uint16Matrix1D | None = None,
    well_predicted_indices: Uint16Matrix1D | None = None,
) -> ELPDMetrics:
    """Compute ELPD metrics and identify poorly / well predicted observations.

    Args:
      elpd_data: ``ELPDData`` from ``arviz_stats``.
      posterior_sigma: Posterior draws of ``sigma``.
      trouble_obs_indices: Indices of poorly predicted observations.
        Auto-detected if ``None``.
      well_predicted_indices: Indices of the best-predicted observations.
        Auto-detected if ``None``.

    Returns:
      A dataclass with attributes ``pointwise_densities``,
      ``avg_loo_log_density``, ``geom_mean_density``,
      ``analytical_log_density_dist``, ``ideal_mean_density``, ``deviation``,
      ``trouble_obs_indices``, ``well_predicted_indices``, and ``obs_to_plot``.

    """
    pointwise_densities: Float64Matrix1D = np.exp(elpd_data.elpd_i.to_numpy())

    avg_loo_log_density = elpd_data.elpd / elpd_data.n_data_points
    geom_mean_density = np.exp(avg_loo_log_density)

    analytical_log_density_dist: Float64Matrix1D = (
        -0.5 * np.log(2 * np.pi) - 0.5 - np.log(posterior_sigma)
    )  # (chain * draw, )
    ideal_mean_density: np.float64 = np.exp(analytical_log_density_dist.mean())
    deviation = np.sqrt(
        2
        * (
            (-0.5 * np.log(2 * np.pi * posterior_sigma.mean() ** 2))
            - avg_loo_log_density
        )
    )

    if trouble_obs_indices is None:
        trouble_obs_indices = cast(
            Uint16Matrix1D,
            np.where(
                pointwise_densities
                < (pointwise_densities.mean() - 2 * pointwise_densities.std())
            )[0],
        )

    if well_predicted_indices is None:
        well_predicted_indices = cast(
            Uint16Matrix1D, np.argsort(pointwise_densities)[-6:]
        )

    print(
        "Poorly predicted observations (abnormally low densities): "
        + f"{trouble_obs_indices}"
    )
    print(
        "The top 5 predicted observations (highest densities): "
        + f"{well_predicted_indices}"
    )
    obs_to_plot = np.concatenate([trouble_obs_indices, well_predicted_indices])

    return ELPDMetrics(
        pointwise_densities=pointwise_densities,
        avg_loo_log_density=avg_loo_log_density,
        geom_mean_density=geom_mean_density,
        analytical_log_density_dist=analytical_log_density_dist,
        ideal_mean_density=ideal_mean_density,
        deviation=deviation,
        trouble_obs_indices=trouble_obs_indices,
        well_predicted_indices=well_predicted_indices,
        obs_to_plot=obs_to_plot,
    )


def plot_loo_predictive_single(
    i: uint16,
    y_obs: Float64Matrix1D,
    ppc_flat: Float64Matrix2D,
    weights: Float64Matrix2D,
    pareto_k: Float64Matrix1D,
    pointwise_densities: Float64Matrix1D,
    ax: Axes,
):
    """Plot the LOO posterior predictive distribution for a single observation.

    Computes a weighted KDE of posterior predictive samples using PSIS
    importance weights, overlays the observed value, the LOO mean, and
    the ±1 SD interval, and marks both the PSIS and KDE density
    estimates at the observed point.

    Args:
      i: Index of the observation to plot.
      y_obs: Observed response variable.
      ppc_flat: Posterior predictive samples ``(n_obs, n_samples)``.
      weights: PSIS importance weights ``(n_obs, n_samples)``.
      pareto_k: Pareto k diagnostic for each observation.
      pointwise_densities: Pointwise ELPD contributions.
      ax: Matplotlib axes to draw on.

    Example:
      ```python
      fig, ax = plt.subplots()
      plot_loo_predictive_single(0, y_obs, ppc, w, k, densities, ax=ax)
      ```

    """
    w = weights[i, :]
    samples = ppc_flat[i, :]

    loo_mean = np.average(samples, weights=w)
    loo_std = np.sqrt(np.average((samples - loo_mean) ** 2, weights=w))
    z_score = (y_obs[i] - loo_mean) / loo_std

    kde = stats.gaussian_kde(samples, weights=w)
    x = np.linspace(samples.min(), samples.max(), 300)

    loo_density_at_obs_psis = pointwise_densities[i]
    loo_density_at_obs_kde = kde(y_obs[i])[0]

    _ = ax.plot(x, kde(x), color="steelblue", lw=2, label="LOO predictive")
    _ = ax.axvline(
        y_obs[i],
        color="red",
        linestyle="--",
        lw=2,
        label=f"Observed = {y_obs[i]:.2f}",
    )
    _ = ax.axvline(
        loo_mean,
        color="steelblue",
        linestyle=":",
        lw=1.5,
        label=f"LOO mean = {loo_mean:.2f}",
    )

    x_shade = np.linspace(loo_mean - loo_std, loo_mean + loo_std, 300)
    _ = ax.fill_between(
        x_shade, kde(x_shade), alpha=0.15, color="steelblue", label="±1 SD"
    )

    _ = ax.scatter(
        [y_obs[i]],
        [loo_density_at_obs_psis],
        color="red",
        zorder=5,
        label=f"PSIS density = {loo_density_at_obs_psis:.2f}",
    )
    _ = ax.scatter(
        [y_obs[i]],
        [loo_density_at_obs_kde],
        color="orange",
        zorder=5,
        marker="x",
        s=80,
        label=f"KDE density = {loo_density_at_obs_kde:.2f}",
    )

    _ = ax.set_title(
        f"Obs {i} | k={pareto_k[i]:.2f} | z={z_score:.2f}",
        color="red" if abs(z_score) > 1 else "black",
    )
    _ = ax.set_xlabel("y")
    _ = ax.set_ylabel("LOO predictive density")
    _ = ax.legend(fontsize=8)


def _plot_elpd_diagnostics(
    metrics: ELPDMetrics,
    y_obs: Float64Matrix1D,
    ppc_flat: Float64Matrix2D,
    weights: Float64Matrix2D,
    pareto_k: Float64Matrix1D,
    ncols_loo: int = 3,
):
    """Build the multi-panel ELPD diagnostic figure.

    Args:
      metrics: dataclass returned by ``_compute_elpd_metrics``.
      y_obs: Observed response variable.
      ppc_flat: Posterior predictive samples ``(n_obs, n_samples)``.
      weights: Importance sampling weights ``(n_obs, n_samples)``.
      pareto_k: Pareto k diagnostic values ``(n_obs,)``.
      ncols_loo: Number of columns in the LOO grid. Defaults to ``3``.

    Returns:
      A matplotlib ``Figure`` with the ELPD diagnostic plots.

    """
    nrows_loo = int(np.ceil(len(metrics.obs_to_plot) / ncols_loo))

    fig = plt.figure(figsize=(25, 8 + 4 * nrows_loo))
    outer_gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, nrows_loo], hspace=0.5
    )
    inner_gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[0], wspace=0.3
    )
    _ = fig.add_subplot(inner_gs_top[0])
    _ = fig.add_subplot(inner_gs_top[1])
    inner_gs_loo = gridspec.GridSpecFromSubplotSpec(
        nrows_loo, ncols_loo, subplot_spec=outer_gs[1], wspace=0.3, hspace=0.1
    )
    for idx in range(len(metrics.obs_to_plot)):
        _ = fig.add_subplot(inner_gs_loo[idx])

    # --- Plot 1: histogram ---
    _ = fig.axes[0].hist(metrics.analytical_log_density_dist)
    _ = fig.axes[0].axvline(
        metrics.analytical_log_density_dist.mean(),
        color="black",
        label="expected log-density under perfect calibration",
    )
    _ = fig.axes[0].axvline(
        metrics.avg_loo_log_density,
        color="red",
        label="actual average out-of-sample predictive performance",
    )
    _ = fig.axes[0].set_title(
        "Distribution of the ideal log-density across posterior draws of sigma \n"
        + f"root-mean-square standardized residual {metrics.deviation:.3f}",
        fontsize=14,
    )
    _ = fig.axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        framealpha=0.4,
        columnspacing=0.8,
        handlelength=1.2,
    )

    # --- Plot 2: scatter ---
    _ = fig.axes[1].scatter(y_obs, metrics.pointwise_densities)
    _ = fig.axes[1].axhline(
        metrics.geom_mean_density,
        color="red",
        linestyle="--",
        label=f"Geometric mean density = {metrics.geom_mean_density:.2f}",
    )
    _ = fig.axes[1].axhline(
        metrics.ideal_mean_density,
        color="black",
        linestyle="--",
        label=f"expected density under perfect calibration = {metrics.ideal_mean_density:.2f}",
    )
    _ = fig.axes[1].set_xlabel("Observed Value")
    _ = fig.axes[1].set_ylabel("Pointwise ELPD contribution")
    _ = fig.axes[1].set_title("Pointwise ELPD contributions for each observation")
    _ = fig.axes[1].set_yscale("log")
    _ = fig.axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=12,
        framealpha=0.4,
        columnspacing=0.8,
        handlelength=1.2,
    )

    # --- Plot 3: LOO predictive grid ---
    for ax, i in zip(fig.axes[2:], metrics.obs_to_plot, strict=False):
        plot_loo_predictive_single(
            i, y_obs, ppc_flat, weights, pareto_k, metrics.pointwise_densities, ax=ax
        )
    _ = fig.suptitle("Model ELPD Evaluation", fontsize=16, y=1.01)

    return fig


def evaluate_model_elpd(
    y_obs: Float64Matrix1D,
    elpd_data: ELPDData,
    posterior_sigma: Float64Matrix1D,
    ppc_flat: Float64Matrix2D,
    weights: Float64Matrix2D,
    pareto_k: Float64Matrix1D,
    ncols_loo: int = 3,
):
    """Evaluate model ELPD with diagnostic plots and LOO predictive checks.

    Computes the average out-of-sample predictive performance (ELPD),
    compares it against the ideal log-density under a perfectly calibrated
    normal model, and identifies poorly predicted observations. Produces a
    multi-panel figure with a histogram of ideal log-densities, a scatter
    of pointwise ELPD contributions, and a grid of LOO predictive
    distributions for the best and worst predicted observations.

    Args:
      y_obs: Observed response variable.
      elpd_data: ``ELPDData`` from ``arviz_stats`` containing ``elpd``,
        ``elpd_i``, and ``n_data_points``.
      posterior_sigma: Posterior draws of the noise standard deviation
        ``sigma`` ``(n_samples,)``.
      ppc_flat: Posterior predictive samples stacked as
        ``(n_obs, n_samples)``.
      weights: Importance sampling weights stacked as
        ``(n_obs, n_samples)``.
      pareto_k: Pareto k diagnostic values for each observation ``(n_obs,)``.
      ncols_loo: Number of columns in the LOO predictive grid. Defaults
        to ``3``.

    Returns:
      A tuple ``(fig, trouble_obs_indices)``, where ``fig`` is the
      matplotlib figure and ``trouble_obs_indices`` are the indices of
      poorly predicted observations that were plotted.

    Example:
      ```python
      fig, bad_idx = evaluate_model_elpd(
          y_obs=y,
          elpd_data=elpd_data,
          posterior_sigma=posterior_sigma,
          ppc_flat=ppc_flat,
          weights=weights,
          pareto_k=pareto_k,
      )
      ```

    """
    metrics = _compute_elpd_metrics(elpd_data, posterior_sigma)
    fig = _plot_elpd_diagnostics(metrics, y_obs, ppc_flat, weights, pareto_k, ncols_loo)
    plt.show()
    return fig, metrics.trouble_obs_indices


# def binomial_band(n: int, grid: np.ndarray, alpha: float = 0.05):
#     lower = stats.binom.ppf(alpha / 2, n, grid) / n
#     upper = stats.binom.ppf(1 - alpha / 2, n, grid) / n
#     return lower, upper


def null_coverage_band(
    weights: Float64Matrix2D,
    grid: Float64Matrix1D,
    rng: np.random.Generator,
    ci_level: float = 0.95,
    B: int = 10000,
) -> tuple[Float64Matrix1D, Float64Matrix1D]:
    """Compute a simulation-based null coverage band under perfect calibration.

    Simulates what the LOO calibration curve would look like if the model were
    perfectly calibrated (true PIT ~ Uniform(0, 1)), while accounting for
    finite-sample estimation noise in the LOO-PIT values. Observations with
    unstable importance sampling weights (high Pareto-k) contribute wider noise
    via lower effective sample size (ESS), naturally widening the band where
    LOO-PIT estimates are least reliable.

    This serves as the reference band in the calibration plot: if the empirical
    coverage curve falls outside this band, it cannot be explained by sampling
    variation alone.

    This is the Bayesian non-parameteric alternative to frequentist binomial
    approach:
    ```python lower = stats.binom.ppf(alpha / 2, n, grid) / n upper =
    stats.binom.ppf(1 - alpha / 2, n, grid) / n
    ```

    Args:
        weights: Normalized Importance sampling weights from LOO approximation,
            one row per observation. Shape ``(n_obs, n_samples)``.
        grid: Array of expected coverage levels (quantile grid) at which to
            evaluate the null band. Typically ``np.linspace(0.05, 0.95, 19)``
            or similar. Shape ``(m,)``.
        rng: NumPy random generator instance for reproducibility. Create via
            ``np.random.default_rng(seed)``.
        ci_level: Coverage level for the null band, e.g. 0.95 produces a 95%
            band. Defaults to 0.95.
        B: Number of simulation draws used to estimate the null band. Higher
            values give smoother bands at the cost of memory and compute.
            Defaults to 10000.

    Returns:
        A tuple ``(lower, upper)`` where each element is an array of shape
        ``(m,)`` corresponding to the lower and upper bounds of the null
        coverage band at each grid point.

    Notes:
        Estimation noise for each observation is modelled as ``Beta(ESS * p, ESS
        * (1 - p))`` centered on the true simulated PIT value ``p``, where ESS
        is derived from the IS weights. This matches the variance of a
        Binomial(ESS, p) estimator (i.e., ``p*(1-p)/ESS``) approximately (i.e.,
        ``p*(1-p)/(ESS+1)``. For observations with very low ESS (k > 0.7
        Pareto), the Beta approaches Uniform(0, 1), contributing maximum
        uncertainty to the band.

        Beta is the natural choice here as it is the conjugate prior for the
        Bernouli/Binomial and lives on [0, 1] and naturally models uncertainty
        about a probability. A noisy PIT value is exactly that — an uncertain
        estimate of something that should be Uniform(0,1). Also, It's
        parameterizable by a "concentration" — the sum ``a + b = ESS * p + ESS *
        (1-p) = ESS`` directly controls how tightly the distribution
        concentrates around ``p``. High ESS → tight; low ESS → diffuse. No other
        bounded distribution gives this as cleanly.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> grid = np.linspace(0.05, 0.95, 19)
        >>> lower, upper = null_coverage_band(weights=is_weights, grid=grid, , rng=rng)

    """
    n: int = weights.shape[0]
    # assure weights are normalized
    weights /= np.sum(weights, axis=1, keepdims=True)
    ess = cast(Float64Matrix1D, 1.0 / np.sum(weights**2, axis=1))  # shape (n_obs,)

    # True PITs under null: (B, n_obs)
    true_pit = rng.uniform(0, 1, size=(B, n))

    # Noisy PIT: Beta(ess*p, ess*(1-p)) centered on true_pit
    # Beta parameters: (B, n_obs)
    # Clamp to avoid Beta(0,x) edge cases
    a = np.clip(ess[None, :] * true_pit, 0.1, None)
    b = np.clip(ess[None, :] * (1 - true_pit), 0.1, None)
    noisy_pit = rng.beta(a, b)

    # Coverage curves: (B, m) where m = grid.siz
    # noisy_pit: (B, n_obs, 1) <= grid: (1, 1, m)
    curves = cast(
        Float64Matrix2D, (noisy_pit[:, :, None] <= grid[None, None, :]).mean(axis=1)
    )  # shape (B, 2)
    alpha = 1 - ci_level
    lower = np.percentile(curves, 100 * alpha / 2, axis=0)
    upper = np.percentile(curves, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def bayesian_bootstrap_band(
    loo_pit: Float64Matrix1D,
    grid: Float64Matrix1D,
    rng: np.random.Generator,
    ci_level: float = 0.95,
    B: int = 10000,
) -> tuple[Float64Matrix1D, Float64Matrix1D]:
    """Compute Bayesian bootstrap uncertainty bands for a LOO calibration curve.

    Estimates posterior uncertainty in the empirical calibration
    curve using the Bayesian bootstrap (Dirichlet reweighting).

    Args:
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

    Returns:
        A tuple ``(lower, upper)`` where each element is an array of shape
        ``(m,)`` corresponding to the lower and upper bounds of Bayesian
        bootsrap band at each grid point.

    Notes:
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
        This method is fully model-agnostic and does not assume any particular
        likelihood (Gaussian, Student-t, etc.).

    """
    loo_pit = np.asarray(loo_pit)
    n = loo_pit.size

    # Step 1: Indicator matrix (m * n) where m = grid.size
    # I[j, i] = 1 if loo_pit[i] <= grid[j]
    indicator = (loo_pit[None, :] <= grid[:, None]).astype(float)

    # Step 2: Dirichlet weights (B * n)
    W = rng.dirichlet(np.ones(n), size=B)

    # Step 3: Compute all bootstrap curves at once where each element is a
    # weighted sample proportion
    # curves_{b,j} = sum_{i=1}^n w_b,i * 1(PIT_i <= q_j)
    # (B * n) @ (n * m) = (B * m)
    curves = W @ indicator.T

    # Step 4: Percentile bands
    alpha = 1 - ci_level
    lower = np.percentile(curves, 100 * alpha / 2, axis=0)
    upper = np.percentile(curves, 100 * (1 - alpha / 2), axis=0)

    return lower, upper


def bayesian_bootstrap_rmse(
    y_true: Float64Matrix1D,
    y_pred: Float64Matrix1D,
    rng: np.random.Generator | None = None,
    n_boot: int = 100000,
) -> Float64Matrix1D:
    """Bayesian bootstrap to estimate the posterior distribution of RMSE.

    Treats each observation as a category in a discrete uniform
    distribution, then simulates the posterior of the RMSE by reweighting
    the squared errors with Dirichlet-distributed weights. Unlike the
    classical bootstrap which resamples with replacement, the Bayesian
    bootstrap resamples the probability distribution itself, producing
    smoother posterior estimates.

    Args:
        y_true: Ground-truth observed values.  Shape ``(n_obs,)``.
        y_pred: Predicted values from the model.  Shape ``(n_obs,)``.
        rng : Random number generator for reproducibility..  Defaults to
            ``None``.
        n_boot: Number of bootstrap replications to generate.  Defaults to
            ``100000``.

    Returns:
        A 1-D array of RMSE values from the Bayesian bootstrap posterior,
        shape ``(n_boot,)``.

    Example:
        ```python
        rmse_posterior = bayesian_bootstrap_rmse(y_true, y_pred, random_seed=42)
        rmse_ci = np.percentile(rmse_posterior, [2.5, 97.5])
        ```

    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(y_true)

    residuals = y_true - y_pred
    squared_residuals = residuals**2

    weights = rng.dirichlet(np.ones(n), size=n_boot)  # (n_boot, n)

    weighted_mse = np.sum(weights * squared_residuals, axis=1)  # (n_boot,)

    return np.sqrt(weighted_mse)


def compute_loo_pit_model_agnostic(
    y_obs: Float64Matrix1D, y_pred_flat: Float64Matrix2D, weights: Float64Matrix2D
) -> Float64Matrix1D:
    """Compute model-agnostic LOO Probability Integral Transform (PIT) values.

    Computes leave-one-out (LOO) PIT values using importance sampling weights
    obtained from PSIS-LOO. It is fully model-agnostic and does not assume any
    specific likelihood (Gaussian, Student-t, etc.).

    Args:
        y_obs: Observed response values, shape (n_obs,).
        y_pred_flat: Posterior predictive draws for each observation, shape
            (n_obs, n_samples), flattened across chains and draws. Each row
            corresponds to one observation and each column to one posterior
            sample.
        weights: Normalized PSIS-LOO importance weights, shape (n_obs,
            n_samples). Each row sums to 1 and approximates p(θ | y_-i) via:
                w_i^(s) ∝ 1 / p(y_i | θ^(s))

    Returns:
        LOO-PIT values for each observation, shape (n_obs,):
            PIT_i = P(y_i^rep ≤ y_i | y_-i)
        computed as a weighted empirical CDF of posterior predictive draws.

    Note:
        For each observation i, the LOO predictive distribution is approximated
        as a weighted empirical measure:
            PIT_i = sigma_s w_i^(s) * 1{ y_i^(s) ≤ y_i }
        where w_i^(s) are PSIS weights and y_i^(s) are predictive draws.

        Under correct model specification, loo_pit ~ Uniform(0, 1). Deviations
        indicate predictive miscalibration: an S-shaped ECDF suggests under- or
        over-dispersion; skewed deviations suggest asymmetric miscalibration.

        This approach is fully model-agnostic, requires no analytic CDF, and
        uses PSIS-LOO without refitting the model.

    """
    # assure weights are normalized
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_obs_reshaped = y_obs[:, np.newaxis]  # Shape (n_obs, 1)
    indicators = (y_pred_flat <= y_obs_reshaped).astype(np.uint16)
    return np.clip(np.sum(weights * indicators, axis=1), 0.0, 1.0)


@dataclass
class CalibrationStats:
    expected_coverage: Float64Matrix1D
    empirical_coverage: Float64Matrix1D
    bootstrap_lower: Float64Matrix1D
    bootstrap_upper: Float64Matrix1D
    reference_lower: Float64Matrix1D
    reference_upper: Float64Matrix1D
    calibration_error: np.float64
    weighted_cal_error: np.float64
    n_miscalibrated: np.uint16


def calculate_empirical_coverage(
    loo_pit: Float64Matrix1D, expected_coverage: Float64Matrix1D
) -> Float64Matrix1D:
    """Compute empirical coverage at each expected-coverage threshold.

    Args:
      loo_pit: Leave-one-out probability integral transform values.
      expected_coverage: Array of nominal coverage levels to evaluate.

    Returns:
      Array of empirical coverage proportions, one per threshold.

    Example:
      ```python
      loo_pit = np.array([0.1, 0.4, 0.6, 0.9])
      expected = np.array([0.25, 0.50, 0.75])
      calculate_empirical_coverage(loo_pit, expected)
      ```

    """
    return np.array([(loo_pit <= q).mean() for q in expected_coverage])


def calculate_calibration_error(
    expected_coverage: Float64Matrix1D, empirical_coverage: Float64Matrix1D
):
    """Evaluate calibration error and variance-weighted calibration error.

    The weighted variant up-weights deviations where the expected coverage
    is near 0.5 (highest variance) and down-weights the extremes.

    Args:
      expected_coverage: Nominal coverage levels.
      empirical_coverage: Observed coverage at each level.

    Returns:
      A tuple ``(calibration_error, weighted_cal_error)``, where
      ``calibration_error`` is the mean absolute deviation and
      ``weighted_cal_error`` is the deviation weighted by
      ``expected_coverage * (1 - expected_coverage)``.

    Example:
      ```python
      expected = np.array([0.25, 0.50, 0.75])
      empirical = np.array([0.20, 0.55, 0.70])
      calculate_calibration_error(expected, empirical)
      ```

    """
    calibration_error: np.float64 = np.mean(
        np.abs(empirical_coverage - expected_coverage), dtype=np.float64
    )
    # Using Bernoulli variance `p(1-p)` as weights up-weights mid-range coverage
    # levels (near 0.5) where sampling variability is highest, and down-weights
    # extremes.
    variance_weights = expected_coverage * (1 - expected_coverage)
    weighted_cal_error: np.float64 = cast(
        np.float64,
        np.average(
            np.abs(empirical_coverage - expected_coverage), weights=variance_weights
        ),
    )
    return calibration_error, weighted_cal_error


def calculate_miscalibrated_coverage(
    empirical_coverage: Float64Matrix1D,
    expected_coverage: Float64Matrix1D,
    sampling_lower: Float64Matrix1D,
    sampling_upper: Float64Matrix1D,
    bootstrap_lower: Float64Matrix1D,
    bootstrap_upper: Float64Matrix1D,
):
    """Identify coverage levels where calibration bands do not overlap.

    A level is considered calibrated when the sampling interval and
    bootstrap interval overlap; otherwise it is miscalibrated.

    Args:
      empirical_coverage: Observed coverage at each level.
      expected_coverage: Nominal coverage at each level.
      sampling_lower: Lower bound of the sampling uncertainty band.
      sampling_upper: Upper bound of the sampling uncertainty band.
      bootstrap_lower: Lower bound of the bootstrap uncertainty band.
      bootstrap_upper: Upper bound of the bootstrap uncertainty band.

    Returns:
      Boolean array where ``True`` indicates a miscalibrated level.

    Example:
      ```python
      empirical = np.array([0.20, 0.55, 0.70])
      expected = np.array([0.25, 0.50, 0.75])
      samp_low = np.array([0.18, 0.52, 0.67])
      samp_high = np.array([0.22, 0.58, 0.73])
      boot_low = np.array([0.15, 0.48, 0.65])
      boot_high = np.array([0.26, 0.53, 0.78])
      calculate_miscalibrated_coverage(
          empirical, expected, samp_low, samp_high, boot_low, boot_high
      )
      ```

    """
    calibrated = (
        (empirical_coverage >= expected_coverage) & (sampling_upper >= bootstrap_lower)
    ) | (
        (empirical_coverage <= expected_coverage) & (sampling_lower <= bootstrap_upper)
    )
    return ~calibrated


def _compute_calibration_curve_data(
    y_obs: Float64Matrix1D,
    y_pred: Float64Matrix2D,
    log_likelihood: Float64Matrix2D | None,
    weights: Float64Matrix2D | None,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> CalibrationCurveParams:
    """Compute LOO calibration curve data, uncertainty bands, and diagnostics.

    Args:
      y_obs: Observed data vector.
      y_pred: Posterior predictive draws of shape (n_obs, n_samples).
      log_likelihood: Log-likelihood matrix of shape (n_obs, n_samples).
      weights: PSIS weights of shape (n_obs, n_samples).
      n_boot: Number of Bayesian bootstrap replications.
      ci_level: Credible interval level for bands.
      rng: NumPy random generator.

    Returns:
      A dict with keys ``expected_coverage``, ``empirical_coverage``,
      ``sampling_lower``, ``sampling_upper``, ``bootstrap_lower``,
      ``bootstrap_upper``, ``calibration_error``, ``weighted_cal_error``,
      ``miscalibrated``, ``n_miscalibrated``, and ``n_obs``.

    """
    expected_coverage: Float64Matrix1D = np.array(
        [*np.arange(0.05, 0.96, 0.05).tolist(), 0.99, 1.0]
    )

    if log_likelihood is not None:
        weights, _pareto_k = compute_psis_weights(log_likelihood)
    assert weights is not None, "PSIS weights are not available!"
    # compute loo pit values
    loo_pit = compute_loo_pit_model_agnostic(y_obs, y_pred, weights)
    # calculate emprirical coverage values
    empirical_coverage = calculate_empirical_coverage(loo_pit, expected_coverage)
    # compute finite-sample uncertainty band
    sampling_lower, sampling_upper = null_coverage_band(
        weights=weights, grid=expected_coverage, rng=rng
    )
    # compute posterior uncertainty
    bootstrap_lower, bootstrap_upper = bayesian_bootstrap_band(
        loo_pit, expected_coverage, rng, ci_level=ci_level, B=n_boot
    )
    # calculate calibration error values
    calibration_error, weighted_cal_error = calculate_calibration_error(
        expected_coverage, empirical_coverage
    )
    # find miscalibrated intervals
    miscalibrated = calculate_miscalibrated_coverage(
        empirical_coverage,
        expected_coverage,
        sampling_lower,
        sampling_upper,
        bootstrap_lower,
        bootstrap_upper,
    )
    n_miscalibrated = np.sum(miscalibrated).astype(np.uint16)

    return {
        "expected_coverage": expected_coverage,
        "empirical_coverage": empirical_coverage,
        "sampling_lower": sampling_lower,
        "sampling_upper": sampling_upper,
        "bootstrap_lower": bootstrap_lower,
        "bootstrap_upper": bootstrap_upper,
        "calibration_error": calibration_error,
        "weighted_cal_error": weighted_cal_error,
        "miscalibrated": miscalibrated,
        "n_miscalibrated": n_miscalibrated,
        "n_obs": y_obs.size,
    }


def plot_loo_calibration_curve_with_reference(
    y_obs: Float64Matrix1D,
    y_pred: Float64Matrix2D,
    *,
    log_likelihood: Float64Matrix2D | None = None,
    weights: Float64Matrix2D | None = None,
    n_boot: int = 10000,
    ci_level: float = 0.95,
    figsize: tuple[float, float] = (7, 7),
    random_seed: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | SubFigure, CalibrationStats]:
    """Plot a LOO calibration curve with reference and bootstrap uncertainty bands.

    Computes the leave-one-out probability integral transform (LOO-PIT) from
    the provided observations and posterior draws, then plots the empirical
    coverage against expected coverage alongside finite-sampling and Bayesian
    bootstrap uncertainty bands. Flagged points indicate significant
    miscalibration where the diagonal falls outside both uncertainty sources
    simultaneously.

    Args:
        y_obs: Observed data vector of shape (n_obs,).
        y_pred: Posterior predictive draws of shape (n_obs, n_samples).
        log_likelihood: Log-likelihood matrix of shape (n_obs, n_samples).
        weights: PSIS weights of shape (n_obs, n_samples).
        n_boot: Number of Bayesian bootstrap replications. Defaults to 10000.
        ci_level: Credible interval level for the bootstrap and sampling bands.
            Defaults to 0.95.
        figsize: Matplotlib figure size as (width, height). Defaults to (7, 7).
        random_seed: Seed for the random number generator. Defaults to None.
        ax: Matplotlib axes to plot on. If None, a new figure and axes are
            created. Defaults to None.

    Returns:
        A tuple (fig, st), where fig is the matplotlib Figure or SubFigure
        instance, and stats is a CalibrationStats named tuple containing
        expected and empirical coverage curves, uncertainty band bounds, and
        calibration error metrics.

    Example:
        >>> fig, st = plot_loo_calibration_curve_with_reference(
        ...     y_obs=y_obs,
        ...     y_pred=y_pred,
        ...     log_likelihood=log_likelihood,
        ...     n_boot=5000,
        ... )
        >>> print(st.calibration_error)

    """
    rng: np.random.Generator = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )

    d = _compute_calibration_curve_data(
        y_obs, y_pred, log_likelihood, weights, n_boot, ci_level, rng
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_aspect("equal")

    _ = ax.plot(
        [0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration", alpha=0.7
    )
    _ = ax.fill_between(
        d["expected_coverage"],
        d["sampling_lower"],
        d["sampling_upper"],
        alpha=0.3,
        color="gray",
        label=f"{int(ci_level * 100)}% Expected sampling variation",
    )
    _ = ax.fill_between(
        d["expected_coverage"],
        d["bootstrap_lower"],
        d["bootstrap_upper"],
        alpha=0.3,
        color="steelblue",
        label=f"{int(ci_level * 100)}% Bayesian Bootstrap",
    )
    _ = ax.plot(
        d["expected_coverage"],
        d["empirical_coverage"],
        "o-",
        linewidth=2.5,
        markersize=7,
        label="LOO Calibration",
        color="steelblue",
        zorder=10,
    )
    # d["miscalibrated"] is a boolean mask (0/1 per coverage level). When used
    # as an index, it picks only the entries where the value is truthy (1);
    # i.e., miscalibrated.
    if d["n_miscalibrated"] > 0:
        mask = d["miscalibrated"]
        _ = ax.scatter(
            d["expected_coverage"][mask],
            d["empirical_coverage"][mask],
            s=150,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            zorder=11,
            label="Significantly miscalibrated",
        )
    _ = ax.set_xlabel("Expected Coverage (HDI Level)", fontsize=13)
    _ = ax.set_ylabel("Empirical Coverage (LOO)", fontsize=13)
    _ = ax.set_title(
        f"""LOO-Based Calibration Curve (n={d["n_obs"]})\n"""
        + f"""calibration error {d["calibration_error"]:.3f}, """
        + f"""weighted calibration error {d["weighted_cal_error"]:.3f}""",
        fontsize=15,
        fontweight="bold",
    )
    _ = ax.legend(fontsize=11, loc="upper left")
    _ = ax.grid(True)
    _ = ax.set_xlim(-0.02, 1.02)
    _ = ax.set_ylim(-0.02, 1.02)
    _ = ax.set_aspect("equal")
    _ = ax.text(
        0.98,
        0.02,
        f"n = {d['n_obs']}\nBayesian bootstrap = {n_boot:,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
    )

    return fig, CalibrationStats(
        expected_coverage=d["expected_coverage"],
        empirical_coverage=d["empirical_coverage"],
        bootstrap_lower=d["bootstrap_lower"],
        bootstrap_upper=d["bootstrap_upper"],
        reference_lower=d["sampling_lower"],
        reference_upper=d["sampling_upper"],
        calibration_error=d["calibration_error"],
        weighted_cal_error=d["weighted_cal_error"],
        n_miscalibrated=d["n_miscalibrated"],
    )


def _prepare_calibration_data(
    idata: DataTree,
) -> tuple[Float64Matrix1D, Float64Matrix2D, Float64Matrix2D]:
    """Extract observations, posterior predictive, and PSIS weights from a DataTree.

    Determines the observation dimension automatically, pulls observed
    data and posterior predictive samples, then computes PSIS importance
    weights from the log-likelihood.

    Args:
      idata: ArviZ ``DataTree`` containing ``observed_data``,
        ``posterior_predictive``, and ``log_likelihood`` groups.

    Returns:
      A tuple ``(y_obs, y_pred, weights)``, where ``y_obs`` is
      ``(n_obs,)``, ``y_pred`` is ``(n_obs, n_samples)``, and
      ``weights`` are normalized PSIS weights ``(n_obs, n_samples)``.

    Example:
      ```python
      y_obs, y_pred, weights = _prepare_calibration_data(idata)
      ```

    """
    obs_dim = next(iter(idata.observed_data.dims.keys()))
    y_obs: Float64Matrix1D = extract(
        idata, group="observed_data", sample_dims=obs_dim
    ).to_numpy()
    y_pred: Float64Matrix2D = extract(
        idata, group="posterior_predictive", combined=True
    ).to_numpy()
    log_lik_flat: Float64Matrix2D = extract(
        idata, group="log_likelihood", combined=True
    ).to_numpy()
    return y_obs, y_pred, log_lik_flat


def plot_loo_calibration_curves(
    idata: DataTree,
    random_seed: int | None = None,
    n_boot: int = 10000,
    ci_level: float = 0.95,
):
    """Plot a side-by-side LOO-PIT envelope and LOO calibration curve.

    Renders the LOO-PIT QQ plot from ``arviz_plots`` alongside a custom
    calibration curve with bootstrap and sampling uncertainty bands.

    Args:
      idata: ArviZ ``DataTree`` containing ``observed_data``,
        ``posterior_predictive``, and ``log_likelihood`` groups.
      random_seed: Seed for reproducible Bayesian bootstrap. Defaults to
        ``None``.
      n_boot: Number of Bayesian bootstrap replications. Defaults to
        ``10000``.
      ci_level: Credible interval level for uncertainty bands. Defaults
        to ``0.95``.

    Returns:
      A tuple ``(fig, calib_res)``, where ``fig`` is the matplotlib
      figure and ``calib_res`` is a ``CalibrationStats`` named tuple.

    Example:
      ```python
      fig, calib = plot_loo_calibration_curves(idata, random_seed=42)
      ```

    """
    pc = azp.plot_loo_pit(
        idata, envelope_prob=ci_level, figure_kwargs={"figsize": (7, 4)}
    )
    loo_pit_img = _render_pc_to_img(pc)
    pc = azp.plot_loo_pit(
        idata, coverage=True, envelope_prob=ci_level, figure_kwargs={"figsize": (7, 4)}
    )
    loo_eti_img = _render_pc_to_img(pc)

    _, (ax_loo_1, ax_loo_2, ax_custom) = plt.subplots(1, 3, figsize=(14, 7))

    ax_loo_1.imshow(loo_pit_img)
    ax_loo_1.axis("off")

    ax_loo_2.imshow(loo_eti_img)
    ax_loo_2.axis("off")

    y_obs, y_pred, log_lik_flat = _prepare_calibration_data(idata)

    fig, calib_res = plot_loo_calibration_curve_with_reference(
        y_obs,
        y_pred,
        log_likelihood=log_lik_flat,
        n_boot=n_boot,
        ci_level=ci_level,
        random_seed=random_seed,
        ax=ax_custom,
    )

    plt.tight_layout()
    plt.show()
    return fig, calib_res


def evaluate_noise_model(
    model_data: BayesianModelData,
    trace_data: MCMCSamples,
    raw_std: dict[int, np.float64],
    random_seed: int,
    hdi_prob: float = 0.95,
):
    """Evaluate noise model calibration by comparing three sigma estimates.

    Compares what the model believes about observation noise (posterior
    sigma) against two empirical residual estimates per setpoint, to detect
    under- or over-confidence in the noise specification.

    The three sigma estimates are:
      1. Model sigma:  The model's explicit posterior over sigma.
      2. Per-draw residual sigma:  std(y - mu_draw) for each posterior draw,
         preserving uncertainty in mu (always >= plug-in by Jensen).
      3. Plug-in residual sigma:  std(y - mu_hat) where mu_hat = E[mu|data],
         the classic frequentist residual SD with bootstrap CI.

    A well-calibrated model has model sigma approximately equal to plug-in
    residual sigma.  If model sigma is too low the model is overconfident;
    if too high it is conservative.  The per-draw estimate additionally shows
    how much mu uncertainty inflates the implied noise.

    Also computes variance decomposition (between- vs within-setpoint) and
    per-setpoint R-squared values to give context for the noise comparison.

    Args:
        model_data: Prepared Bayesian model data containing the standardized
            response (``y_scaled``), standardization moments (``y_std``,
            ``y_mean``), and setpoint labels (``setpoint_timeseries``).
        trace_data: MCMC samples containing the posterior mean
            (``mu_raw``, shape ``(chain, draw, obs)``), the noise
            parameter (``sigma_raw``, shape ``(chain, draw[, obs])``), and
            observed values (``y_obs``, shape ``(obs,)``).
        raw_std: Raw standard deviation of the data keyed by setpoint.
        random_seed: Seed for the Bayesian bootstrap reproducibility.
        hdi_prob: Highest-density-interval probability for summaries.
            Defaults to 0.95.

    Returns:
        A dict with model_sigma_summary, residual_sigma_per_draw_summary,
        plugin_sigma (per-setpoint point estimate), plugin_sigma_ci
        (bootstrap CI), r2_overall, r2_within_setpoint (per-setpoint),
        variance_decomposition (total, between, within, percentages), and
        fig (the matplotlib figure).

    Example:
        ```python
        result = evaluate_noise_model(
            model_data=model_data,
            trace_data=samples,
            raw_std={60: 0.12, 80: 0.15},
            random_seed=42,
        )
        ```

    """
    setpoint_unique = np.unique(model_data.setpoint_timeseries)

    perf = _compute_overall_performance(
        trace_data.y_obs.to_numpy(),
        model_data.y_std,
        model_data.y_mean,
        trace_data.mu_raw,
    )
    var_decomp = _compute_variance_decomposition(
        perf.y_original, model_data.setpoint_timeseries, setpoint_unique
    )
    model_sigma_by_sp = _extract_model_sigma(
        trace_data.sigma_raw, setpoint_unique, model_data.y_std
    )
    residual_per_draw = _compute_drawwise_residual_sigma(
        trace_data.mu_raw,
        trace_data.y_obs.to_numpy(),
        model_data.setpoint_timeseries,
        setpoint_unique,
        model_data.y_std,
    )
    plugin = _compute_plugin_statistics(
        trace_data.y_obs.to_numpy(),
        model_data,
        perf,
        setpoint_unique,
        random_seed,
        hdi_prob,
    )

    model_sigma_summary = {
        sp: _summarize_draws(d, hdi_prob) for sp, d in model_sigma_by_sp.items()
    }
    residual_sigma_per_draw_summary = {
        sp: _summarize_draws(d, hdi_prob) for sp, d in residual_per_draw.items()
    }

    fig = _create_noise_diagnostic_figure(
        setpoint_unique,
        model_sigma_summary,
        plugin,
        residual_sigma_per_draw_summary,
        var_decomp,
        perf.r2_overall,
        raw_std,
        hdi_prob,
    )

    _print_diagnostics(
        setpoint_unique,
        model_sigma_summary,
        plugin,
        raw_std,
        var_decomp,
        perf.r2_overall,
    )

    return {
        "model_sigma_summary": model_sigma_summary,
        "residual_sigma_per_draw_summary": residual_sigma_per_draw_summary,
        "plugin_sigma": plugin.plugin_sigma,
        "plugin_sigma_ci": plugin.plugin_sigma_ci,
        "r2_overall": perf.r2_overall,
        "r2_within_setpoint": plugin.within_setpoint_r2,
        "variance_decomposition": {
            "total": var_decomp.var_total,
            "between_setpoint": var_decomp.var_between,
            "within_setpoint": var_decomp.var_within,
            "pct_between": var_decomp.var_between / var_decomp.var_total * 100,
            "pct_within": var_decomp.var_within / var_decomp.var_total * 100,
        },
        "fig": fig,
    }


# =============================================================================
# Internal helpers for evaluate_noise_model
# =============================================================================


def _compute_overall_performance(
    y_true: Float64Matrix1D,
    y_std: float,
    y_mean: float,
    mu_posterior: DataArray,
) -> PerformanceResult:
    """Compute posterior mean, back-transform to original scale, and overall R².

    Args:
        y_true: Observed response values (standardized).  Shape ``(obs,)``.
        y_std: Standard deviation used to standardize the target.
        y_mean: Mean used to standardize the target.
        mu_posterior: Posterior samples of the mean function.  Shape
            ``(chain, draw, obs)``.

    Returns:
        A ``PerformanceResult`` with the posterior mean estimate, the
        original-scale arrays, and the overall coefficient of determination.

    Example:
        ```python
        perf = _compute_overall_performance(y_true, 0.5, 10.0, mu_samples)
        ```

    """
    mu_hat = mu_posterior.mean(axis=(0, 1)).to_numpy()  # (n_obs, )
    y_original = y_true * y_std + y_mean
    mu_hat_original = mu_hat * y_std + y_mean
    ss_tot = np.sum((y_original - y_original.mean()) ** 2)
    ss_res = np.sum((y_original - mu_hat_original) ** 2)
    r2_overall = 1.0 - ss_res / ss_tot
    return PerformanceResult(mu_hat, y_original, mu_hat_original, r2_overall)


def _compute_variance_decomposition(
    y_original: Float64Matrix1D,
    setpoint_timeseries: pd.Series,
    setpoint_unique: NDArray[np.uint16],
) -> VarDecompResult:
    """Decompose total variance into between- and within-setpoint components.

    Uses the ANOVA identity: total = between-group + within-group (pooled).
    Between-setpoint variance measures how much the setpoint means differ
    from the grand mean.  Within-setpoint variance is the noise remaining
    after accounting for setpoint.

    Args:
        y_original: Response values on the original (unstandardized) scale.
            Shape ``(obs,)``.
        setpoint_timeseries: Setpoint label for each observation.  Shape
            ``(obs,)``.
        setpoint_unique: Sorted unique setpoint values.

    Returns:
        A ``VarDecompResult`` containing total, between-setpoint, and
        within-setpoint (pooled) variance.

    Example:
        ```python
        vd = _compute_variance_decomposition(
            y, sp_timeseries, np.unique(sp_timeseries)
        )
        ```

    """
    var_total = cast(np.float64, np.var(y_original))
    grand_mean = y_original.mean()
    n_per_sp = {sp: (setpoint_timeseries == sp).sum() for sp in setpoint_unique}
    n_total = len(y_original)
    var_between = (
        sum(
            n_per_sp[sp]
            * (y_original[setpoint_timeseries == sp].mean() - grand_mean) ** 2
            for sp in setpoint_unique
        )
        / n_total
    )
    var_within = (
        sum(
            n_per_sp[sp] * np.var(y_original[setpoint_timeseries == sp])
            for sp in setpoint_unique
        )
        / n_total
    )
    return VarDecompResult(var_total, var_between, var_within)


def _extract_model_sigma(
    sigma_posterior: DataArray,
    setpoint_unique: NDArray[np.uint16],
    y_std: float,
) -> dict[np.uint16, Float64Matrix1D]:
    """Extract the model's posterior sigma values scaled to the original units.

    Handles two structures transparently: a single global sigma (2-D array)
    or a per-setpoint sigma (3-D array).  For the global case the same
    posterior draws are broadcast to every setpoint entry.

    Args:
        sigma_posterior: Posterior samples of the noise parameter.  Shape
            ``(chain, draw)`` for a global sigma or ``(chain, draw,
            setpoint)`` for per-setpoint sigma.
        setpoint_unique: Sorted unique setpoint values.
        y_std: Standard deviation used to standardize the target; the
            posterior draws are multiplied by this to return original-scale
            standard deviations.

    Returns:
        A dict mapping each setpoint to a 1-D array of posterior sigma
        draws (``n_chain * n_draw`` elements), all on the original scale.

    Example:
        ```python
        sigma_by_sp = _extract_model_sigma(sigma_samples, np.unique(sp), 0.5)
        ```

    """
    if len(sigma_posterior.dims) == 2:  # (chain, draw) — global sigma
        sigma_flat = sigma_posterior.to_numpy().flatten() * y_std
        return dict.fromkeys(setpoint_unique, sigma_flat)
    # (chain, draw, setpoint) — per-setpoint sigma
    return dict(
        zip(
            setpoint_unique,
            sigma_posterior.mean(dim=("setpoint")).to_numpy() * y_std,
            strict=False,
        )
    )


def _compute_drawwise_residual_sigma(
    mu_posterior: DataArray,
    y_true: Float64Matrix1D,
    setpoint_timeseries: pd.Series,
    setpoint_unique: NDArray[np.uint16],
    y_std: float,
) -> dict[np.uint16, Float64Matrix1D]:
    """Compute residual standard deviation for each posterior draw of mu.

    For every MCMC draw of the mean function, this computes
    ``std(y - mu_draw)`` within each setpoint.  The result is a posterior
    over the residual sigma that retains uncertainty in mu, in contrast to
    the plug-in estimate which conditions on ``E[mu|data]``.

    Args:
        mu_posterior: Posterior samples of the mean function.  Shape
            ``(chain, draw, obs)``.
        y_true: Observed response values (standardized).  Shape ``(obs,)``.
        setpoint_timeseries: Setpoint label for each observation.  Shape
            ``(obs,)``.
        setpoint_unique: Sorted unique setpoint values.
        y_std: Standard deviation used to standardize the target; results
            are multiplied by this to return original-scale values.

    Returns:
        A dict mapping each setpoint to a 1-D array of residual sigma
        draws (``n_chain * n_draw`` elements) on the original scale.

    Example:
        ```python
        res_sigma = _compute_drawwise_residual_sigma(
            mu_samples, y, sp, np.unique(sp), 0.5
        )
        ```

    """
    residuals_per_draw = y_true[None, None, :] - mu_posterior
    result = {}
    for sp in setpoint_unique:
        mask = (setpoint_timeseries == sp).to_numpy()
        sigma_draws = (
            residuals_per_draw[:, :, mask].std(axis=2).to_numpy().flatten() * y_std
        )
        result[sp] = sigma_draws  # (n_samples,)
    return result


def _compute_plugin_statistics(
    y_true: Float64Matrix1D,
    model_data: BayesianModelData,
    performance: PerformanceResult,
    setpoint_unique: NDArray[np.uint16],
    random_seed: int,
    hdi_prob: float,
) -> PluginResult:
    """Compute plug-in residual sigma, its bootstrap CI, and within-setpoint R².

    The plug-in residual uses a point estimate of the mean (``mu_hat = E[mu
    | data]``), analogous to a frequentist regression.  A Bayesian bootstrap
    provides an uncertainty interval around the residual standard deviation.
    Within-setpoint R² is also computed per setpoint.

    Args:
        y_true: Observed response values (standardized).  Shape ``(obs,)``.
        model_data: Prepared Bayesian model data providing
            ``setpoint_timeseries`` and ``y_std`` for grouping and
            rescaling.
        performance: Pre-computed overall performance statistics
            containing ``mu_hat``, ``y_original``, and
            ``mu_hat_original`` on the original scale.
        setpoint_unique: Sorted unique setpoint values.
        random_seed: Seed for the Bayesian bootstrap reproducibility.
        hdi_prob: Highest-density-interval probability for the bootstrap
            CI of the plug-in sigma estimate.  Defaults to 0.95.

    Returns:
        A ``PluginResult`` containing the per-setpoint plug-in sigma, its
        bootstrap CI, and the within-setpoint R².

    Example:
        ```python
        plugin = _compute_plugin_statistics(
            y_true, model_data, perf, np.unique(sp), 42
        )
        ```

    """
    residuals_plugin = y_true - performance.mu_hat
    plugin_sigma: dict[np.uint16, float] = {}
    plugin_sigma_ci: dict[np.uint16, np.ndarray] = {}
    within_setpoint_r2: dict[np.uint16, float] = {}
    for sp in setpoint_unique:
        mask = (model_data.setpoint_timeseries == sp).to_numpy()
        res_sp = residuals_plugin[mask]
        plugin_sigma[sp] = float(res_sp.std() * model_data.y_std)
        ci = hdi(
            bayesian_bootstrap_sigma(np.asarray(res_sp), random_seed), prob=hdi_prob
        )
        plugin_sigma_ci[sp] = ci * model_data.y_std
        y_sp = performance.y_original[mask]
        mu_sp = performance.mu_hat_original[mask]
        ss_tot_sp = np.sum((y_sp - y_sp.mean()) ** 2)
        ss_res_sp = np.sum((y_sp - mu_sp) ** 2)
        within_setpoint_r2[sp] = 1 - ss_res_sp / ss_tot_sp if ss_tot_sp > 0 else 0.0
    return PluginResult(plugin_sigma, plugin_sigma_ci, within_setpoint_r2)


def _create_noise_diagnostic_figure(
    setpoint_unique: NDArray[np.uint16],
    model_sigma_summary: dict[np.uint16, dict[str, float]],
    plugin: PluginResult,
    residual_sigma_per_draw_summary: dict[np.uint16, dict[str, float]],
    var_decomp: VarDecompResult,
    r2_overall: float,
    raw_std: dict[int, np.float64],
    hdi_prob: float,
) -> Figure:
    """Build the 4-panel diagnostic figure comparing noise and variance estimates.

    Panel layout:
      - Top-left: model sigma vs plug-in residual sigma (primary calibration
        check).
      - Top-right: model sigma vs per-draw residual sigma (shows effect of
        mu uncertainty).
      - Bottom-left: variance decomposition showing raw SD, residual SD, and
        model sigma across setpoints.
      - Bottom-right: bar chart of total, between-setpoint, and
        within-setpoint variance with overall R² annotation.

    Args:
        setpoint_unique: Sorted unique setpoint values.
        model_sigma_summary: Summarised posterior of model sigma per
            setpoint (mean, median, HDI).
        plugin: Plug-in residual sigma statistics per setpoint.
        residual_sigma_per_draw_summary: Summarised per-draw residual sigma
            per setpoint.
        var_decomp: Variance decomposition results.
        r2_overall: Overall coefficient of determination.
        raw_std: Raw standard deviation keyed by setpoint.
        hdi_prob: Highest-density-interval probability used for the HDI
            bands in the plots.

    Returns:
        The matplotlib figure with the four diagnostic panels.

    Example:
        ```python
        fig = _create_noise_diagnostic_figure(
            sp_unique,
            model_summary,
            plugin,
            resid_summary,
            vd,
            r2,
            raw_std,
            0.95,
        )
        ```

    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _plot_model_vs_plugin(
        axs[0, 0],
        setpoint_unique,
        model_sigma_summary,
        plugin.plugin_sigma,
        plugin.plugin_sigma_ci,
        hdi_prob,
        raw_std,
    )
    _plot_model_vs_perdraw(
        axs[0, 1],
        setpoint_unique,
        model_sigma_summary,
        residual_sigma_per_draw_summary,
        hdi_prob,
        raw_std,
    )
    _plot_variance_decomposition(
        axs[1, 0], setpoint_unique, raw_std, plugin.plugin_sigma, model_sigma_summary
    )
    _plot_variance_breakdown(
        axs[1, 1],
        var_decomp.var_total,
        var_decomp.var_between,
        var_decomp.var_within,
        r2_overall,
    )
    plt.tight_layout()
    return fig


def bayesian_bootstrap_sigma(
    y: Float64Matrix1D, random_seed: int | None = None, n_boot: int = 100000
) -> np.ndarray:
    """Bayesian bootstrap to estimate the confidence interval of sigma.

    Treats the observed data points as if they came from a discrete uniform
    distribution, then simulates the posterior distribution of any statistic
    (here, the standard deviation) by reweighting the data. Each bootstrap draw
    represents a different plausible empirical distribution consistent with the
    observed data. We are not resampling points but resampling distributions.

    Args:
        y : The data points from which to estimate the standard deviation, shape
        (n_samples,)
        random_seed: Seed for reproducible Bayesian bootstrap. Defaults to
        ``None``.
        n_boot : The number of bootstrap samples to generate. Default is 10,000.

    Returns:
        A tuple containing the lower and upper bounds of the 95% confidence
            interval for the standard deviation.

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


def _summarize_draws(draws: Float64Matrix1D, prob: float):  # (n_samples,)
    """Summarize a 1-D array of posterior draws with mean, median, and HDI.

    Args:
        draws: Posterior draws of a scalar quantity.  Shape ``(n_samples,)``.
        prob: Highest-density-interval probability.  Defaults to 0.95.

    Returns:
        A dict with ``mean``, ``median``, ``hdi_lower``, and
        ``hdi_upper`` keys.

    Example:
        ```python
        summary = _summarize_draws(sigma_draws, prob=0.95)
        ```

    """
    hdi_value = hdi(draws, prob=prob)
    return {
        "mean": float(np.mean(draws)),
        "median": float(np.median(draws)),
        "hdi_lower": float(hdi_value[0]),
        "hdi_upper": float(hdi_value[1]),
    }


def _plot_model_vs_plugin(
    ax: Axes,
    setpoints: NDArray[np.uint16],
    model_summary: dict[np.uint16, dict[str, float]],
    plugin_sigma: dict[np.uint16, float],
    plugin_ci: dict[np.uint16, np.ndarray],
    hdi_prob: float,
    raw_std: dict[int, np.float64],
):
    """Plot model sigma against plug-in residual sigma as the primary calibration check.

    Displays the model's posterior sigma (HDI band + mean line) alongside
    the plug-in residual sigma points with bootstrap confidence intervals
    and the raw sigma reference line.  A well-calibrated model has the
    black residual-sigma points falling inside the blue HDI band.

    Args:
        ax: Matplotlib axes to draw on.
        setpoints: Sorted unique setpoint values for the x-axis.
        model_summary: Summarised model sigma per setpoint (must contain
            ``mean``, ``hdi_lower``, ``hdi_upper``).
        plugin_sigma: Plug-in residual sigma per setpoint (point estimate).
        plugin_ci: Bootstrap CI for the plug-in sigma per setpoint.  Each
            entry is a 2-element array ``[lower, upper]``.
        hdi_prob: Probability used for the HDI band (shown in the legend
            label).
        raw_std: Raw standard deviation keyed by setpoint (reference line).

    Example:
        ```python
        _plot_model_vs_plugin(
            ax, sp, model_summary, plugin_sigma, plugin_ci, 0.95, raw_std
        )
        ```

    """
    # Model posterior
    means = [model_summary[sp]["mean"] for sp in setpoints]
    lower = [model_summary[sp]["hdi_lower"] for sp in setpoints]
    upper = [model_summary[sp]["hdi_upper"] for sp in setpoints]

    _ = ax.fill_between(
        setpoints,
        lower,
        upper,
        alpha=0.3,
        color="C0",
        label=f"Model sigma ({int(hdi_prob * 100)}% HDI)",
    )
    _ = ax.plot(
        setpoints, means, "o-", color="C0", label="Model sigma (posterior mean)"
    )

    # Plug-in residual sigma with CI
    plugin_vals = [plugin_sigma[sp] for sp in setpoints]
    for sp in setpoints:
        lo, hi = plugin_ci[sp]
        _ = ax.vlines(sp, lo, hi, color="black", linewidth=2.5, alpha=0.7)
    _ = ax.scatter(
        setpoints,
        plugin_vals,
        color="black",
        s=70,
        zorder=5,
        label="Residual sigma at μ̂ (plug-in)",
    )
    _ = ax.plot(
        setpoints,
        [raw_std[sp] for sp in setpoints],
        linestyle=":",
        color="gray",
        alpha=0.5,
        label="Raw sigma (reference)",
    )
    _ = ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    _ = ax.set_ylabel("Standard Deviation")
    _ = ax.set_title(
        "Primary Check: Model sigma vs Observed Residuals\n"
        + "(Black points should fall within blue band)",
        fontsize=11,
    )
    _ = ax.legend(loc="best")
    _ = ax.grid(alpha=0.3)


def _plot_model_vs_perdraw(
    ax: Axes,
    setpoints: NDArray[np.uint16],
    model_summary: dict[np.uint16, dict[str, float]],
    perdraw_summary: dict[np.uint16, dict[str, float]],
    hdi_prob: float,
    raw_std: dict[int, np.float64],
):
    """Plot model sigma against draw-wise residual sigma (includes mu uncertainty).

    Overlays the model's posterior sigma (blue) with the per-draw residual
    sigma (orange).  The orange band is expected to be wider because it
    propagates uncertainty in the mean function mu.  The raw sigma
    reference line is shown in gray.

    Args:
        ax: Matplotlib axes to draw on.
        setpoints: Sorted unique setpoint values for the x-axis.
        model_summary: Summarised model sigma per setpoint.
        perdraw_summary: Summarised per-draw residual sigma per setpoint
            (must contain ``mean``, ``hdi_lower``, ``hdi_upper``).
        hdi_prob: Probability used for the HDI bands (shown in legend
            labels).
        raw_std: Raw standard deviation keyed by setpoint (reference line).

    Example:
        ```python
        _plot_model_vs_perdraw(
            ax, sp, model_summary, perdraw_summary, 0.95, raw_std
        )
        ```

    """
    # Model sigma
    model_means = [model_summary[sp]["mean"] for sp in setpoints]
    model_lower = [model_summary[sp]["hdi_lower"] for sp in setpoints]
    model_upper = [model_summary[sp]["hdi_upper"] for sp in setpoints]

    _ = ax.fill_between(
        setpoints,
        model_lower,
        model_upper,
        alpha=0.3,
        color="C0",
        label=f"Model sigma ({int(hdi_prob * 100)}% HDI)",
    )
    _ = ax.plot(
        setpoints, model_means, "o-", color="C0", label="Model sigma (posterior mean)"
    )

    # Per-draw residual sigma
    res_means = [perdraw_summary[sp]["mean"] for sp in setpoints]
    res_lower = [perdraw_summary[sp]["hdi_lower"] for sp in setpoints]
    res_upper = [perdraw_summary[sp]["hdi_upper"] for sp in setpoints]

    _ = ax.fill_between(
        setpoints,
        res_lower,
        res_upper,
        alpha=0.3,
        color="C1",
        label=f"Residual sigma per draw ({int(hdi_prob * 100)}% HDI)",
    )
    _ = ax.plot(
        setpoints, res_means, "s--", color="C1", label="Residual sigma per draw"
    )
    _ = ax.plot(
        setpoints,
        [raw_std[sp] for sp in setpoints],
        linestyle=":",
        color="gray",
        alpha=0.5,
        label="Raw sigma (reference)",
    )

    _ = ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    _ = ax.set_ylabel("Standard Deviation")
    _ = ax.set_title(
        "Model sigma vs Draw-wise Residual sigma\n"
        + "(Orange includes μ uncertainty → wider)",
        fontsize=11,
    )
    _ = ax.legend(loc="best")
    _ = ax.grid(alpha=0.3)


def _plot_variance_decomposition(
    ax: Axes,
    setpoints: NDArray[np.uint16],
    raw_std: dict[int, np.float64],
    plugin_sigma: dict[np.uint16, float],
    model_summary: dict[np.uint16, dict[str, float]],
):
    """Plot raw SD, residual SD, and model sigma to visualise explained variance.

    Three lines are drawn per setpoint: raw SD (total variability),
    residual SD (unexplained), and model sigma (estimated noise).  The
    green shaded region between the raw SD and residual SD lines
    represents the variance explained by the mean function.

    Args:
        ax: Matplotlib axes to draw on.
        setpoints: Sorted unique setpoint values for the x-axis.
        raw_std: Raw standard deviation keyed by setpoint.
        plugin_sigma: Plug-in residual sigma per setpoint.
        model_summary: Summarised model sigma per setpoint (uses ``mean``
            for the line).

    Example:
        ```python
        _plot_variance_decomposition(
            ax, sp, raw_std, plugin_sigma, model_summary
        )
        ```

    """
    raw_vals = [raw_std[sp] for sp in setpoints]
    residual_vals = [plugin_sigma[sp] for sp in setpoints]
    model_vals = [model_summary[sp]["mean"] for sp in setpoints]

    _ = ax.plot(
        setpoints,
        raw_vals,
        "o--",
        color="gray",
        label="Raw SD (total variability)",
        linewidth=2,
    )
    _ = ax.plot(
        setpoints,
        residual_vals,
        "o-",
        color="black",
        label="Residual SD (unexplained)",
        linewidth=2,
    )
    _ = ax.plot(
        setpoints,
        model_vals,
        "o-",
        color="C0",
        label="Model sigma (estimated noise)",
        linewidth=2,
    )

    # Shade the "explained" region
    _ = ax.fill_between(
        setpoints,
        residual_vals,
        raw_vals,
        alpha=0.2,
        color="green",
        label="Variance explained by μ",
    )

    _ = ax.set_xlabel("Outlet Pressure Setpoint (psi)")
    _ = ax.set_ylabel("Standard Deviation")
    _ = ax.set_title(
        "Variance Decomposition\n(Gap = variance explained by mean function)",
        fontsize=11,
    )
    _ = ax.legend(loc="best")
    ax.grid(alpha=0.3)


def _plot_variance_breakdown(
    ax: Axes, var_total: float, var_between: float, var_within: float, r2_overall: float
):
    """Plot a bar chart of the variance decomposition with percentage labels.

    Shows total, between-setpoint, and within-setpoint variance as a
    grouped bar chart.  Percentage labels indicate the proportion each
    component contributes to the total.  An annotation highlights how
    much variance the model captures via the between-setpoint effect.

    Args:
        ax: Matplotlib axes to draw on.
        var_total: Total variance of the response.
        var_between: Between-setpoint variance component.
        var_within: Within-setpoint (pooled) variance component.
        r2_overall: Overall coefficient of determination, shown in the
            title.

    Example:
        ```python
        _plot_variance_breakdown(ax, 10.0, 7.5, 2.5, 0.75)
        ```

    """
    # Bar chart showing decomposition
    categories = ["Total\nVariance", "Between\nSetpoint", "Within\nSetpoint"]
    values = [var_total, var_between, var_within]
    colors = ["steelblue", "seagreen", "coral"]

    _ = ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.7)

    # Add percentage labels
    _ = ax.text(
        1,
        var_between,
        f"{var_between / var_total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    _ = ax.text(
        2,
        var_within,
        f"{var_within / var_total * 100:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    _ = ax.set_ylabel("Variance")
    _ = ax.set_title(
        f"Variance Decomposition\n(Overall R² = {r2_overall:.1%})", fontsize=11
    )
    ax.grid(alpha=0.3, axis="y")

    # Add annotation
    _ = ax.annotate(
        f"Model captures {var_between / var_total * 100:.1f}% of variance\n"
        + "(between-setpoint effect)",
        xy=(1, var_between),
        xytext=(1.5, var_total * 0.7),
        arrowprops={"arrowstyle": "->", "color": "gray"},
        fontsize=9,
        ha="left",
    )


def _print_diagnostics(
    setpoints: NDArray[np.uint16],
    model_summary: dict[np.uint16, dict[str, float]],
    plugin: PluginResult,
    raw_std: dict[int, np.float64],
    var_decomp: VarDecompResult,
    r2_overall: float,
):
    """Print a formatted diagnostic table and interpretation to the console.

    Outputs three sections:
      1. Variance decomposition summary (total, between, within).
      2. Per-setpoint table with model sigma, residual sigma, their
         ratio, within-setpoint R², and raw SD.
      3. Interpretation verdict describing whether the model is well
         calibrated, overconfident, or conservative.

    Args:
        setpoints: Sorted unique setpoint values.
        model_summary: Summarised model sigma per setpoint.
        plugin: Plug-in residual sigma results per setpoint.
        raw_std: Raw standard deviation keyed by setpoint.
        var_decomp: Variance decomposition results.
        r2_overall: Overall coefficient of determination.

    Example:
        ```python
        _print_diagnostics(sp, model_summary, plugin, raw_std, var_decomp, 0.85)
        ```

    """
    print("=" * 75)
    print("NOISE MODEL CALIBRATION DIAGNOSTICS")
    print("=" * 75)

    # Variance decomposition summary
    print("\n📊 VARIANCE DECOMPOSITION:")
    print("-" * 75)
    print(f"  Total variance:           {var_decomp.var_total:>10.2f}")
    print(
        f"  Between-setpoint:         {var_decomp.var_between:>10.2f}  "
        + f"({var_decomp.var_between / var_decomp.var_total * 100:>5.1f}%)"
    )
    print(
        f"  Within-setpoint (pooled): {var_decomp.var_within:>10.2f}  "
        + f"({var_decomp.var_within / var_decomp.var_total * 100:>5.1f}%)"
    )
    print(
        f"\n  Overall R²: {r2_overall:.3f} (model captures between-setpoint variation)"
    )

    # Per-setpoint table
    print("\n📋 PER-SETPOINT DIAGNOSTICS:")
    print("-" * 75)
    print(
        f"{'Setpoint':>10} | {'Model sigma':>9} | {'Residual sigma':>11} | "
        + f"{'Ratio':>6} | {'Within R²':>9} | {'Raw SD':>8}"
    )
    print("-" * 75)

    ratios = []
    for sp in setpoints:
        model_s = model_summary[sp]["mean"]
        resid_s = plugin.plugin_sigma[sp]
        ratio = resid_s / model_s
        ratios.append(ratio)
        r2_sp = plugin.within_setpoint_r2[sp]
        raw_s = raw_std[sp]

        # Flag interpretation
        flag = ""
        if ratio > 1.1:
            flag = " ⚠️"
        elif ratio < 0.9:
            flag = " 📉"

        print(
            f"{sp:>10.0f} | {model_s:>9.3f} | {resid_s:>11.3f} | "
            + f"{ratio:>6.2f}{flag} | {r2_sp:>8.1%} | {raw_s:>8.3f}"
        )

    print("-" * 75)
    print(
        f"{'Mean':>10} | {model_summary[setpoints[0]]['mean']:>9.3f} | "
        + f"{np.mean(list(plugin.plugin_sigma.values())):>11.3f} | {np.mean(ratios):>6.2f} |"
    )

    # Interpretation
    print("\n" + "=" * 75)
    print("📖 INTERPRETATION:")
    print("=" * 75)
    print(f"""
    Your model has R² = {r2_overall:.1%} overall, but low within-setpoint R².

    This is EXPECTED and CORRECT because:

    1. {var_decomp.var_between / var_decomp.var_total * 100:.1f}% of total variance is BETWEEN setpoints
       → Model mean function (μ) captures this well ✓

    2. {var_decomp.var_within / var_decomp.var_total * 100:.1f}% of total variance is WITHIN setpoints
       → This is mostly irreducible noise (sensor noise, minor fluctuations)
       → Model sigma parameter captures this ✓

    The noise model evaluation above is checking whether model sigma correctly
    estimates this within-setpoint noise — and with mean ratio = {np.mean(ratios):.2f},
    it does!

    ┌─────────────────────────────────────────────────────────────────────┐
    │  VERDICT: Well-specified model                                      │
    │  • Mean function captures systematic variation (between setpoints)  │
    │  • Sigma captures residual noise (within setpoints)                 │
    │  • Slight conservative bias (ratio < 1) is acceptable               │
    └─────────────────────────────────────────────────────────────────────┘
    """)


@dataclass(frozen=True)
class ModelVars:
    """Variable names used in the PyMC model."""

    like_var: str = "y_like"
    mu_var: str = "mu"
    sigma_var: str = "sigma"
    nu_var: str = "nu"


_MODEL_VARS_DEFAULT = ModelVars()


def evaluate_model_performance(
    idata: DataTree,
    trace_data: MCMCSamples,
    setpoint_timeseries: pd.Series,
    model_vars: ModelVars = _MODEL_VARS_DEFAULT,
    likehood: LikeLiHood = LikeLiHood.NORMAL,
    hdi_prob: float = 0.95,
    random_seed: int | None = None,
    figsize: tuple[int, int] = (10, 5),
):
    """Evaluate Bayesian model performance and produce diagnostic plots.

    Computes a comprehensive suite of performance metrics including Bayesian
    R-squared, MAE, RMSE, and LOO-adjusted variants, then generates diagnostic
    plots for MAE distributions, residuals vs fitted values, and autocorrelation
    of posterior predictive residuals.

    Args:
      idata: Inference data object containing posterior and posterior predictive.
      trace_data: Stacked MCMC samples for observed and predicted values.
      setpoint_timeseries: Setpoint values indexed by observation time.
      model_vars: Variable names used in the PyMC model. Defaults to
          ``_MODEL_VARS_DEFAULT``.
      likehood: Likelihood distribution type. Defaults to
          ``LikeLiHood.NORMAL``.
      hdi_prob: Highest density interval probability. Defaults to ``0.95``.
      random_seed: Seed for reproducible bootstrap resampling. Defaults to
          ``None``.
      figsize: Width and height of generated figures. Defaults to ``(10, 5)``.

    Returns:
      MetricsResult containing Bayesian R-squared, residual R-squared,
      LOO-adjusted R-squared, MAE (posterior, pointwise, LOO), and RMSE
      (posterior, pointwise, LOO).

    Example:
      ```python
      metrics = evaluate_model_performance(
          idata=idata,
          trace_data=trace_data,
          setpoint_timeseries=setpoint_series,
          model_vars=ModelVars(),
          likehood=LikeLiHood.NORMAL,
          hdi_prob=0.95,
          random_seed=42,
      )
      ```

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    obs_dim = next(iter(idata.observed_data.dims.keys()))
    sample_dims = tuple(
        d for d in idata.posterior[model_vars.mu_var].dims if d != obs_dim
    )
    y_pred, residuals_obs, residuals_draw, metrics = _compute_metrics(
        idata,
        sample_dims,
        obs_dim,
        trace_data,
        hdi_prob,
        model_vars,
        likehood,
        rng,
    )

    draw_autocorrs = np.abs(compute_autocorr(residuals_draw))  # (n_samples)

    _print_metrics(metrics, hdi_prob)
    _plot_mae_distributions(metrics, hdi_prob, figsize)

    hdi_data = idata.posterior_predictive[model_vars.like_var]
    hdi_width = _compute_hdi_width(hdi_data, sample_dims, hdi_prob)
    _plot_residuals(residuals_obs, y_pred, hdi_width, setpoint_timeseries, figsize)

    # picks the draw with the largest lag-1 autocorrelation across observations
    max_autcorr_draw_idx = np.argmax(draw_autocorrs)
    residuals_draw_max_autcorr = residuals_draw[max_autcorr_draw_idx]

    _plot_autocorrelation(
        draw_autocorrs,
        cast(pd.DatetimeIndex, setpoint_timeseries.index),
        residuals_draw_max_autcorr,
    )

    return metrics


# =============================================================================
# Internal helpers for evaluate_model_performance
# =============================================================================


@dataclass
class MetricsResult:
    """Container for computed metrics."""

    bayes_r2: tuple[np.ndarray, ...] | np.ndarray
    residual_r2: tuple[np.ndarray, ...] | np.ndarray
    loo_r2: tuple[np.ndarray, ...] | np.ndarray
    mae_obs: np.ndarray
    rmse_obs: float
    rmse_obs_resampled: Float64Matrix1D
    mae_posterior: Float64Matrix1D
    mae_obs_az: tuple[np.ndarray, ...]
    mae_loo: tuple[np.ndarray, ...]
    rmse_posterior: Float64Matrix1D
    rmse_obs_az: tuple[np.ndarray, ...]
    rmse_loo: tuple[np.ndarray, ...]


def _compute_metrics(
    idata: DataTree,
    sample_dims: tuple[str, str],
    obs_dim: str,
    trace_data: MCMCSamples,
    hdi_prob: float,
    model_vars: ModelVars,
    likehood: LikeLiHood,
    rng: np.random.Generator,
) -> tuple[Float64Matrix2D, Float64Matrix1D, Float64Matrix2D, MetricsResult]:
    """Compute pointwise and posterior performance metrics.

    Calculates observed residuals, per-draw residuals, and a full set of
    Bayesian metrics (R-squared, MAE, RMSE, and LOO-adjusted variants) using
    arviz_stats routines.

    Args:
      idata: Inference data object.
      sample_dims: Dimension names to treat as samples (e.g., ``("chain", "draw")``).
      obs_dim: Observation dimension name.
      trace_data: Stacked MCMC samples.
      hdi_prob: Probability for highest density intervals.
      model_vars: Variable names from the PyMC model.
      likehood: Likelihood distribution type.
      rng: Random number generator for bootstrap resampling.

    Returns:
      A tuple ``(y_pred, residuals_obs, residuals_draw, metrics)`` where
      ``y_pred`` is the mean prediction per observation, ``residuals_obs`` is
      the pointwise residual, ``residuals_draw`` is the per-draw residual
      matrix, and ``metrics`` is a MetricsResult container.

    """
    # trace_data.y_pred_stacked:(n_obs, n_samples)
    y_pred = trace_data.y_pred_stacked.mean(axis=1).to_numpy()  # (n_obs, )
    residuals_obs = (trace_data.y_obs - y_pred).to_numpy()  # (n_obs, )
    # For every draw, compute a full residual vector (one per observation)
    residuals_draw = (
        trace_data.y_obs.to_numpy()[None, :] - trace_data.y_pred_stacked.transpose()
    ).to_numpy()  # (n_samples, n_obs) # pyright: ignore[reportAttributeAccessIssue]
    idata = calculate_modeled_residual_var(idata, model_vars, likehood, obs_dim)

    metrics = MetricsResult(
        bayes_r2=bayesian_r2(
            idata,
            pred_mean=model_vars.mu_var,
            group="posterior",
            summary=True,
            scale="var_residual",
            ci_kind="hdi",
            ci_prob=hdi_prob,
            round_to="4g",
        ),
        residual_r2=residual_r2(
            idata,
            pred_mean=model_vars.mu_var,
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
        rmse_obs_resampled=bayesian_bootstrap_rmse(
            trace_data.y_obs.to_numpy(), y_pred, rng
        ),
        loo_r2=loo_r2(
            idata,
            var_name=model_vars.like_var,
            summary=True,
            round_to="4g",
            ci_kind="hdi",
            ci_prob=0.95,
        ),
        mae_obs_az=azs_metrics(
            idata, kind="mae", sample_dims=sample_dims, round_to="3g"
        ),
        mae_loo=loo_metrics(idata, kind="mae", round_to="3g"),
        rmse_obs_az=azs_metrics(
            idata, kind="rmse", sample_dims=sample_dims, round_to="3g"
        ),
        rmse_loo=loo_metrics(idata, kind="rmse", round_to="3g"),
    )
    return y_pred, residuals_obs, residuals_draw, metrics


def _print_metrics(metrics: MetricsResult, hdi_prob: float) -> None:
    """Print Bayesian performance metrics to stdout.

    Args:
      metrics: Container with computed metrics.
      hdi_prob: Probability used for HDI intervals displayed in output.

    """
    print("Model performance:")
    print(f"Bayesian R²: {metrics.bayes_r2}")
    print(f"Residual R²: {metrics.residual_r2}")
    print(
        f"Bayesian posterior MAE: mean={metrics.mae_posterior.mean():.4f}, "
        + f"95% HDI={hdi(metrics.mae_posterior, prob=hdi_prob)}"
    )
    print(
        f"Bayesian pointwise MAE: mean={metrics.mae_obs.mean():.4f}, "
        + f"95% HDI={hdi(metrics.mae_obs, prob=hdi_prob)}"
    )
    print(f"Arviz built-in MAE: {metrics.mae_obs_az}")
    print(
        f"Bayesian posterior RMSE: mean={metrics.rmse_posterior.mean():.4f}, "
        + f"95% HDI={hdi(metrics.rmse_posterior, prob=hdi_prob)}"
    )
    print(
        f"Bayesian pointwise RMSE: mean={metrics.rmse_obs:.4f}, "
        + f"95% HDI={hdi(metrics.rmse_obs_resampled, prob=hdi_prob)}"
    )
    print(f"Arviz built-in RMSE: {metrics.rmse_obs_az}")
    print(f"Leave-one-out Cross-validation adjusted R^2: {metrics.loo_r2}")
    print(f"Leave-one-out Cross-validation MAE: {metrics.mae_loo}")
    print(f"Leave-one-out Cross-validation RMSE: {metrics.rmse_loo}")


def _plot_mae_distributions(
    metrics: MetricsResult, hdi_prob: float, figsize: tuple[int, int]
) -> None:
    """Plot histograms of per-draw and per-observation MAE.

    Args:
      metrics: Container with MAE posterior and observed values.
      hdi_prob: Probability for the HDI annotation on each subplot.
      figsize: Width and height of the figure.

    """
    _, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].hist(metrics.mae_posterior)
    axs[0].set_xlabel("Mean Absolute Error per Draw")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title(
        "Distribution of MAE across Posterior Draws\nPosterior Uncertainty\n"
        + f"mean={metrics.mae_posterior.mean():.4f}, "
        + f"95% HDI={hdi(metrics.mae_posterior, prob=hdi_prob)}",
        fontsize=10,
    )

    axs[1].hist(metrics.mae_obs)
    axs[1].set_xlabel("Mean Absolute Error per Observation")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title(
        "Distribution of MAE across Observations\nObservation-Level Prediction "
        + f"Uncertainty\nmean={metrics.mae_obs.mean():.4f}, "
        + f"95% HDI={hdi(metrics.mae_obs, prob=hdi_prob)}",
        fontsize=10,
    )
    plt.show()


def _plot_residuals(
    residuals_obs: Float64Matrix1D,
    fitted_values: Float64Matrix2D,
    hdi_width: float,
    setpoint_timeseries: pd.Series,
    figsize: tuple[int, int],
) -> None:
    """Plot residuals vs fitted values with HDI bounds.

    This diagnostic plot helps identify:
    - Non-random patterns (suggests model misspecification)
    - Heteroscedasticity (funnel shape)
    - Outliers
    - Whether residuals are similar across setpoints

    Args:
      residuals_obs: Observed residuals.
      fitted_values: Model fitted or predicted values.
      hdi_width: Width of the highest density interval.
      setpoint_timeseries: Setpoint values for each observation.
      figsize: Width and height of the figure.

    """
    # Create figure
    _, ax = plt.subplots(figsize=figsize)

    # Get unique setpoints for coloring
    setpoints = np.unique(setpoint_timeseries)

    # Plot residuals colored by setpoint
    for sp in setpoints:
        mask = np.isclose(setpoint_timeseries, sp, atol=1e-6)
        _ = ax.scatter(
            fitted_values[mask],
            residuals_obs[mask],
            label=f"SP = {int(sp)}",
            alpha=0.6,
            s=50,
        )

    # Add reference lines
    _ = ax.axhline(
        y=0, color="black", linestyle="-", linewidth=1.0, label="Zero", zorder=1
    )
    _ = ax.axhline(
        y=hdi_width / 2,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="HDI half-width",
        zorder=1,
    )
    _ = ax.axhline(
        y=-hdi_width / 2, color="red", linestyle="--", linewidth=1.5, zorder=1
    )

    # Labels and title
    _ = ax.set_xlabel("Fitted Values", fontsize=12)
    _ = ax.set_ylabel("Residual", fontsize=12)
    _ = ax.set_title("Residuals vs Fitted Values", fontsize=14, fontweight="bold")

    # Legend with your preferred settings
    _ = ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        framealpha=0.95,
        facecolor="white",
        edgecolor="gray",
    )

    # Add grid for easier pattern detection
    ax.grid(True, alpha=0.75, linestyle=":", linewidth=0.5)

    plt.tight_layout()


def _plot_autocorrelation(
    draw_autocorrs: Float64Matrix1D,
    time_series_index: pd.DatetimeIndex,
    residual_draw: Float64Matrix1D,
) -> None:
    """Plot lag-1 autocorrelation distribution and the worst-case ACF.

    Args:
      draw_autocorrs: Lag-1 autocorrelation per posterior draw.
      time_series_index: Datetime index for the time series.
      residual_draw: Residual vector for the draw with the highest lag-1
          autocorrelation.

    """
    _, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].hist(draw_autocorrs)
    axs[0].set_title(
        "Distribution of Lag-1 Autocorrelation Plot across all the posterior draws",
        fontsize=12,
        fontweight="bold",
    )
    _ = pd.plotting.autocorrelation_plot(
        pd.Series(index=time_series_index, data=residual_draw), ax=axs[1]
    )
    _ = axs[1].set_title(
        "Autocorrelation Plot for the posterior draw with maximum Lag-1 autocorrelation",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()


def calculate_modeled_residual_var(
    idata: DataTree,
    model_vars: ModelVars,
    likehood: LikeLiHood = LikeLiHood.NORMAL,
    obs_dim: str = "obs",
):
    """Compute and store residual variance in the posterior DataTree.

    For a Normal likelihood the residual variance is ``sigma`` (or the mean
    across ``obs_dim`` when that dimension exists).  For a Student-t
    likelihood it is the marginal variance
    ``sigma**2 * nu / (nu - 2)``.

    Args:
      idata: Inference data object; receives the computed variance in-place.
      model_vars: Variable names from the PyMC model.
      likehood: Likelihood distribution type. Defaults to
          ``LikeLiHood.NORMAL``.
      obs_dim: Observation dimension name. Defaults to ``"obs"``.

    Returns:
      DataTree with ``var_residual`` added to the ``posterior`` group.

    """
    mu = idata.posterior[model_vars.mu_var]
    sigma = idata.posterior[model_vars.sigma_var]
    has_obs_dim = obs_dim in mu.dims
    has_obs_dim = len(sigma.dims) > 2
    has_obs_dim = obs_dim in mu.dims and len(sigma.dims) > 2
    if likehood.NORMAL:
        if has_obs_dim:
            idata.posterior["var_residual"] = sigma.mean(axis=-1)
        else:
            idata.posterior["var_residual"] = sigma
    elif likehood.T:
        var_residual = (
            idata.posterior[model_vars.sigma_var] ** 2
            * idata.posterior[model_vars.nu_var]
            / (idata.posterior[model_vars.nu_var] - 2)
        )
        idata.posterior["var_residual"] = (
            var_residual.mean(dim=obs_dim) if has_obs_dim else var_residual
        )
    return idata


def compute_autocorr(x: Float64Matrix2D) -> Float64Matrix1D:
    """Compute lag-1 autocorrelation for each row of a 2D array.

    Args:
      x: Matrix of shape ``(n_samples, n_obs)``.

    Returns:
      Array of shape ``(n_samples,)`` with the lag-1 autocorrelation per
      draw computed across all observations.

    """
    x_centered = x - x.mean(axis=1, keepdims=True)
    num = (x_centered[:, :-1] * x_centered[:, 1:]).sum(axis=1)
    denom = np.sqrt(
        (x_centered[:, :-1] ** 2).sum(axis=1) * (x_centered[:, 1:] ** 2).sum(axis=1)
    )
    return num / denom


def _compute_hdi_width(
    hdi_data: DataArray, sample_dims: str | tuple[str, str], hdi_prob: float
) -> float:
    """Compute the mean width of highest density intervals across observations.

    Args:
      hdi_data: Posterior predictive samples.
      sample_dims: Dimension names to marginalise over.
      hdi_prob: Probability mass covered by the HDI.

    Returns:
      Scalar mean HDI width.

    """
    hdi_value = hdi(hdi_data, dim=sample_dims, prob=hdi_prob)
    lower, upper = hdi_value[:, 0], hdi_value[:, 1]
    return np.mean(upper - lower).to_numpy()


def calculate_cusum_with_uncertainty(
    residuals_distribution: Float64Matrix2D,
    timestamps: pd.DatetimeIndex,
    shutin_mask: pd.Series,
    baseline_mask: pd.Series,
    anomaly_direction: Literal["pos", "neg"],
    drift_multiplier: float = 0.5,
    n_samples_subset: int | None = None,
    random_seed: int | None = None,
) -> UncertainCusumResult:
    """Calculate CUSUM paths with posterior uncertainty via Monte Carlo sampling.

    Computes a CUSUM for each posterior sample of residuals, then aggregates
    them into mean, median, confidence intervals, and per-sample drift/mean
    parameters. Shut-in periods force a reset to zero, and a reset occurs on
    the first operational time step after each shut-in.

    Args:
        residuals_distribution: Posterior samples of residuals. Shape
            ``(n_samples, n_obs)``.
        timestamps: Timestamps corresponding to each time point.Shape
            ``(n_obs, )``
        shutin_mask: Boolean mask indicating shut-in periods.Shape
            ``(n_obs, )``
        baseline_mask: Boolean mask indicating the baseline reference period.
            Shape ``(n_obs, )``
        anomaly_direction: Direction of the expected anomaly.
        drift_multiplier: Multiplier ``k`` for the drift term
            ``k * sigma``. Defaults to ``0.5``.
        n_samples_subset: Number of samples to randomly subsample for speed.
            Defaults to ``None`` (all samples are used).
        random_seed: Seed for reproducible subsampling. Defaults to ``None``.

    Returns:
        ``UncertainCusumResult`` containing the full CUSUM paths and summary
        statistics.

    Example:
        ```python
        result = calculate_cusum_with_uncertainty(
            residuals_distribution=posterior_samples,
            timestamps=ts,
            shutin_mask=shutin,
            baseline_mask=baseline,
            anomaly_direction="pos",
        )
        ```

    """
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
    return UncertainCusumResult(
        cusum_paths=cusum_paths,
        mean=cusum_paths.mean(axis=0),
        median=np.median(cusum_paths, axis=0),
        std=cusum_paths.std(axis=0),
        ci_95=np.percentile(cusum_paths, [2.5, 97.5], axis=0),
        ci_80=np.percentile(cusum_paths, [10.0, 90.0], axis=0),
        ci_50=np.percentile(cusum_paths, [25.0, 75.0], axis=0),
        timestamps=timestamps,
        target_means=target_means.flatten(),
        drifts=drifts.flatten(),
        anomaly_direction=anomaly_direction,
    )


def compute_masked_statistics(
    residual_distribution: Float64Matrix2D,
    timestamps: pd.DatetimeIndex,  # (n_obs,)
    shutin_mask: pd.Series,  # (n_obs,)
) -> UncertainResidualStats:
    """Compute per-time-point residual statistics, handling shut-in periods.

    Mean, median, and confidence intervals (50%, 80%, 95%) are computed only
    for operational (non-shut-in) time points. Shut-in entries are filled with
    ``NaN``.

    Args:
        residual_distribution: Posterior samples of residuals. Shape
            ``(n_samples, n_obs)``.
        timestamps: Timestamps of each observation. Shape ``(n_obs, )``
        shutin_mask: Boolean series indicating shut-in periods. Shape ``(n_obs, )``

    Returns:
        ``UncertainResidualStats`` with per-time-point summary statistics and
        an ``operational_mask``.

    Example:
        ```python
        stats = compute_masked_statistics(
            residual_distribution=samples,
            timestamps=ts,
            shutin_mask=shutin,
        )
        ```

    """
    # Align mask
    shutin_aligned = shutin_mask.reindex(timestamps).fillna(False).to_numpy()
    operational_mask = ~shutin_aligned

    _n_samples, n_time = residual_distribution.shape

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

    return UncertainResidualStats(
        mean=residual_mean,
        median=residual_median,
        ci_95=residual_ci_95,
        ci_80=residual_ci_80,
        ci_50=residual_ci_50,
        timestamps=timestamps,
        operational_mask=operational_mask,
    )


def compute_exceedance_probability(
    cusum_paths: Float64Matrix2D, threshold: np.float64
) -> tuple[Float64Matrix1D, Float64Matrix1D, Float64Matrix1D]:
    """Compute the probability of exceeding a threshold at each time point.

    Args:
        cusum_paths: CUSUM paths across posterior samples. Shape
            ``(n_samples, n_obs)``.
        threshold: Control limit threshold ``h``.

    Returns:
        A tuple ``(p_above, p_below, p_either)``, where ``p_above`` is
        ``P(CUSUM > +h)``, ``p_below`` is ``P(CUSUM < -h)``, and
        ``p_either`` is ``P(|CUSUM| > h)``. Each has shape ``(n_obs,)``.

    Example:
        ```python
        p_above, p_below, p_either = compute_exceedance_probability(
            cusum_paths=paths, threshold=3.0
        )
        ```

    """
    p_above = (cusum_paths > threshold).mean(axis=0)  # (n_obs,)
    p_below = (cusum_paths < -threshold).mean(axis=0)  # (n_obs,)
    p_either = (np.abs(cusum_paths) > threshold).mean(axis=0)  # (n_obs,)

    return p_above, p_below, p_either


def visualize_probabilistic_cusum(
    residual_stats: UncertainResidualStats,
    cusum_result: UncertainCusumResult,
    timestamps: pd.DatetimeIndex,
    threshold: np.float64,
    exceed_prob: Float64Matrix1D,
    save: bool = False,
    fname: str | None = None,
):
    """Visualize probabilistic CUSUM results in a multi-panel figure.

    Produces a 4-panel figure showing: (1) CUSUM path with 95% CI,
    (2) exceedance probability over time, (3) residuals with uncertainty,
    and (4) CUSUM uncertainty width. Shut-in periods are shaded gray.

    Args:
        residual_stats: Residual statistics with operational mask.
        cusum_result: CUSUM result with mean, median, and CI arrays.
        timestamps: Timestamps for the x-axis.
        threshold: Control limit threshold ``h``.
        exceed_prob: Exceedance probability at each time point. Shape
            ``(n_obs,)``.
        save: Whether to save the figure to disk. Defaults to ``False``.
        fname: Output filename (required when ``save=True``). Defaults to
            ``None``.

    Raises:
        ValueError: If ``save=True`` and ``fname`` is ``None``.

    Example:
        ```python
        visualize_probabilistic_cusum(
            residual_stats=stats,
            cusum_result=result,
            timestamps=ts,
            threshold=3.0,
            exceed_prob=p_above,
            save=True,
            fname="cusum.png",
        )
        ```

    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

    # timestamps = timestamps
    op_mask = residual_stats.operational_mask

    # ===== Plot 1: CUSUM (already correct) =====
    ax = axes[0]
    ax.fill_between(
        timestamps,
        cusum_result.ci_95[0],
        cusum_result.ci_95[1],
        alpha=0.2,
        color="blue",
        label="95% CI",
    )
    ax.plot(timestamps, cusum_result.mean, "b-", lw=2, label="Mean CUSUM")
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
        residual_stats.ci_95[0],
        residual_stats.ci_95[1],
        alpha=0.3,
        color="blue",
        label="95% CI",
    )
    ax.plot(timestamps, residual_stats.mean, "b-", lw=1, label="Mean residual")
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
    cusum_width = cusum_result.ci_95[1] - cusum_result.ci_95[0]
    # Set shut-in periods to NaN for cleaner plot
    cusum_width_clean = cusum_width.copy()
    cusum_width_clean[~op_mask] = np.nan
    ax.plot(timestamps, cusum_width_clean, "purple", lw=2)
    ax.set_ylabel("95% CI Width")
    ax.set_xlabel("Time")
    ax.set_title("CUSUM Uncertainty (Operational Periods Only)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        if fname is None:
            raise ValueError("fname required when save=True")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    del fig, ax


def compute_coverage(
    y_true: Float64Matrix1D, hdi_intervals: Float64Matrix2D
) -> tuple[np.float64, Float64Matrix1D, Float64Matrix1D]:
    """Compute the fraction of true values falling within HDI intervals.

    Args:
        y_true: Ground truth values. Shape ``(n_obs,)``.
        hdi_intervals: HDI lower and upper bounds. Shape ``(n_obs, 2)``.

    Returns:
        A tuple ``(coverage, lower, upper)``, where ``coverage`` is the
        fraction of ``y_true`` values within their respective intervals,
        and ``lower`` / ``upper`` are the interval bounds.

    Example:
        ```python
        coverage, lo, hi = compute_coverage(y_true, hdi_intervals)
        ```

    """
    lower = hdi_intervals[:, 0]
    upper = hdi_intervals[:, 1]

    within_interval = (y_true >= lower) & (y_true <= upper)
    return cast(np.float64, np.nanmean(within_interval)), lower, upper


def plot_predictions_with_uncertainty(
    data: DataArray,
    y_true: Float64Matrix1D,
    y_pred: Float64Matrix1D,
    hdi_prob: float = 0.95,
    dim: tuple[str, str] = ("chain", "draw"),
):
    """Plot out-of-sample predictions with HDI uncertainty bands.

    Computes HDI intervals from posterior predictive samples, calculates
    coverage and average interval width, and renders a figure with the
    true values, mean predictions, and shaded uncertainty.

    Args:
        data: Posterior predictive samples. Shape ``(chain, draw, obs)``.
        y_true: Ground truth values. Shape ``(n_obs,)``.
        y_pred: Mean point predictions. Shape ``(n_obs,)``.
        hdi_prob: HDI probability mass. Defaults to ``0.95``.
        dim: Names of the chain and draw dimensions in ``data``. Defaults to
            ``("chain", "draw")``.

    Example:
        ```python
        plot_predictions_with_uncertainty(posterior_pred, y_test, y_pred_mean)
        ```

    """
    # for each of the n_obs test data points, we have chain * draw posterior
    # predictive samples, and we want to compute one 95% HDI per test point
    hdi_value = hdi(data, dim=dim, prob=hdi_prob)  # (n_obs, 2)
    coverage, lower, upper = compute_coverage(y_true, hdi_value.to_numpy())
    hdi_width = np.nanmean(
        upper - lower
    )  # average uncertainty window around each prediction
    print("Average HDI width:", hdi_width)
    print(f"{int(hdi_prob * 100)}% HDI coverage:", coverage)
    _ = plt.figure(figsize=(10, 5))
    _ = plt.fill_between(
        np.arange(len(y_true)),
        lower,
        upper,
        alpha=0.3,
        label=f"{hdi_prob * 100}% HDI",
    )
    _ = plt.plot(y_true, label="True", marker="o")
    _ = plt.plot(y_pred, label="Mean Pred")
    _ = plt.legend(loc="upper left")
    _ = plt.title("Out-of-Sample Predictions with Uncertainty")
    plt.show()


def build_mixture_baseline(
    y_deviation: Float64Matrix1D,  # (n_obs, )
    kwargs: dict[str, float],
    n_components: int = 2,
    random_seed: int | None = None,
):
    """Build a Bayesian Gaussian mixture model with ordered component means.

    Constructs a PyMC model that fits a mixture of Gaussians to the observed
    deviations. Component means are ordered via an Ordered transform to prevent
    label switching during MCMC sampling. Dirichlet priors are placed on the
    mixture weights, and half-normal priors on the component standard deviations.

    Args:
        y_deviation: Observed deviation values of shape (n_obs,).
        kwargs: Dictionary of prior hyperparameters containing keys ``mu_mu``,
            ``mu_sigma``, and ``sigma_sigma``.
        n_components: Number of mixture components. Defaults to 2.
        random_seed: Seed for the random number generator. Defaults to None.

    Returns:
        A compiled or un-compiled PyMC model object ready for inference.

    Example:
        ```python
        import numpy as np

        y = np.random.normal(0, 1, 100)
        kwargs = {"mu_mu": 0, "mu_sigma": 2, "sigma_sigma": 1}
        model = build_mixture_baseline(
            y, kwargs, n_components=3, random_seed=42
        )
        ```

    """
    rng = (
        np.random.default_rng(random_seed)
        if random_seed is not None
        else np.random.default_rng()
    )
    coords = {
        "obs_id": np.arange(len(y_deviation)),
        "component": np.arange(n_components),
    }

    with pm.Model(coords=coords) as model:
        # Data
        y_data = pm.Data("y_data", y_deviation, dims="obs_id")
        # --------- Mixture weights ---------
        weights = pm.Dirichlet(
            "weights",
            a=np.ones(n_components),
            shape=(n_components,),
            dims="component",
            rng=rng,
        )
        # --------- Ordered component means (prevents label switching) ---------
        # When values are out of order, the Ordered transform maps them to a
        # space where the log-probability evaluates to \(-\infty \) because the
        # transformation calculates the differences between consecutive elements
        # (\(x_2 - x_1\), \(x_3 - x_2\)). If an element is out of order, this
        # difference becomes negative, which is mathematically invalid in the
        # transformed log-space (as it requires taking the logarithm of a
        # negative number).
        mu = pm.Normal(
            "mu",
            mu=kwargs["mu_mu"],
            sigma=kwargs["mu_sigma"],
            shape=n_components,
            transform=pm.distributions.transforms.Ordered(),
            dims="component",
            rng=rng,
        )
        sigma = pm.HalfNormal(
            "sigma",
            sigma=kwargs["sigma_sigma"],
            shape=n_components,
            dims="component",
            rng=rng,
        )
        # --------- Component distributions ---------
        componets = pm.Normal.dist(mu=mu, sigma=sigma, shape=(n_components,))  # pyright: ignore[reportArgumentType]
        # ---------
        # Mixture likelihood
        # ---------
        _ = pm.Mixture(
            "y_like",
            w=weights,
            comp_dists=componets,
            observed=y_data,
            dims="obs_id",
            rng=rng,
        )

    return model


def evaluate_mixture_model(
    idata: DataTree, y_true: Float64Matrix1D, y_grid: Float64Matrix1D
):
    """Plot the fitted mixture model against empirical data.

    Computes the posterior predictive PDF from the MCMC samples and produces
    a three-panel figure showing the empirical histogram, the fitted mixture
    density, and per-component group membership probabilities.

    Args:
        idata: ArviZ InferenceData object containing ``posterior`` with
            ``mu``, ``sigma``, and ``weights`` dimensions.
        y_true: Observed data values used for the histogram.
        y_grid: Dense 1-D grid over which the PDF is evaluated.

    Returns:
        Matplotlib figure with three vertically stacked axes.

    Example:
        ```python
        import numpy as np

        y = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 200)
        fig = evaluate_mixture_model(idata, y, grid)
        ```

    """
    post = idata.posterior
    pdf_components = (
        XrContinuousRV(stats.norm, post["mu"], post["sigma"]).pdf(y_grid)  # pyright: ignore[reportAttributeAccessIssue]
        * post["weights"]
    )
    pdf = pdf_components.sum("component")
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
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
    plt.tight_layout()
    return fig


def get_mixture_residuals(
    idata: DataTree, y: Float64Matrix1D
) -> MixtureModelResidualStats:
    """Compute residuals and component assignments from a fitted mixture model.

    Extracts posterior means of mixture parameters, computes log-responsibilities
    via softmax, performs a hard assignment to the most likely component, and
    returns both raw and standardized residuals relative to that component.

    Args:
        idata: ArviZ InferenceData object containing ``posterior`` with
            ``mu``, ``sigma``, and ``weights``.
        y: Observed data values of shape (n_obs,).

    Returns:
        A ``MixtureModelResidualStats`` named tuple with fields ``residuals``,
        ``standardized_residuals``, ``assignment``, ``probability``,
        ``predicted_mu``, and ``predicted_sigma``.

    Example:
        ```python
        stats = get_mixture_residuals(idata, y_obs)
        ```

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
        log_prob[:, k] = np.log(weights[k]) + stats.norm.logpdf(y, mu[k], sigma[k])

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

    return MixtureModelResidualStats(
        residuals=residuals,
        standardized_residuals=standardized_residuals,
        assignment=assignment,
        probability=prob,
        predicted_mu=predicted_mu,
        predicted_sigma=predicted_sigma,
    )


def plot_mixture_residuals(residuals_stats: MixtureModelResidualStats):
    """Plot residual diagnostics for a fitted mixture model.

    Produces a side-by-side figure with a Q-Q plot of the standardized
    residuals (testing normality of the residuals) and a bar chart of
    the component assignment counts.

    Args:
        residuals_stats: Result from ``get_mixture_residuals`` containing
            ``standardized_residuals`` and ``assignment`` fields.

    Returns:
        Matplotlib figure with two subplots.

    Example:
        ```python
        stats = get_mixture_residuals(idata, y_obs)
        fig = plot_mixture_residuals(stats)
        ```

    """
    standardized_residuals = residuals_stats.standardized_residuals
    assignment = residuals_stats.assignment

    # Plot residual diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # 1. Q-Q plot (should follow diagonal)

    _ = stats.probplot(
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

    plt.tight_layout()
    plt.show()
    return fig
