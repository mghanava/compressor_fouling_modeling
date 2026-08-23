"""Build a compact m x k summary table from the 6 simulation result parquet files.

Usage:
    python build_summary_table.py

Assumes files are named like: df_results_m_5_k_12_parquet.gzip
and are all in DATA_DIR (edit below). Adjust the regex if your naming differs.
"""

from pathlib import Path
import re
from typing import cast

import pandas as pd

# ---- config -----------------------------------------------------------
# Derive project root from this script's location:
# .../calibration_simulations/calibration_simulation_results_analysis.py
# -> .../calibration_simulations/ -> .../ (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIB_SIM_DIR = _PROJECT_ROOT / "calibration_simulations"
CALIB_SIM_RES_DIR = CALIB_SIM_DIR / "simulation_results"
FILE_GLOB = "df_results_m_*_k_*.parquet.gzip"
ALPHA = 0.05  # significance threshold for rejection-rate metrics
OUT_CSV = Path(CALIB_SIM_RES_DIR / "summary_table_long.csv")
OUT_XLSX = Path(CALIB_SIM_RES_DIR / "summary_table_pivoted.xlsx")
# -------------------------------------------------------------------------

FNAME_RE = re.compile(r"m_(\d+)_k_(\d+)")


_DESIRED_ORDER = [(48, 10), (24, 10), (6, 10), (12, 5), (12, 10), (12, 15)]
_KM_ORDER = {pair: i for i, pair in enumerate(_DESIRED_ORDER)}


def _km_sort_key(k: int, m: int) -> tuple[int, int, int]:
    return _KM_ORDER.get((k, m), len(_KM_ORDER)), k, m


def _sort_key(path: Path) -> tuple[int, int, int]:
    match = FNAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse m/k from filename: {path}")
    m, k = int(match.group(1)), int(match.group(2))
    return _km_sort_key(k, m)


def load_all(data_dir: str, pattern: str) -> pd.DataFrame:
    """Load all matching simulation result files into a single DataFrame.

    Files are sorted by their (k, m) values parsed from the filenames, and
    those values are added as columns to each frame before concatenation.

    Args:
        data_dir: Directory containing the simulation result files.
        pattern: Glob pattern for result filenames, e.g.
            "df_results_m_*_k_*.parquet.gzip".

    Returns:
        A DataFrame with all rows from the matched files, plus "k", "m",
        and "source_file" columns identifying each file's origin.

    Raises:
        FileNotFoundError: No files in data_dir match pattern.
        ValueError: A matched filename does not contain parsable m/k
            values.

    Example:
        ```python
        df_all = load_all(
            "calibration_simulations/simulation_results",
            "df_results_m_*_k_*.parquet.gzip",
        )
        ```

    """
    files = sorted(Path(data_dir).glob(pattern), key=_sort_key)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")

    frames = []
    for f in files:
        match = FNAME_RE.search(f.name)
        if not match:
            raise ValueError(f"Could not parse m/k from filename: {f}")
        m, k = int(match.group(1)), int(match.group(2))

        df = pd.read_parquet(f)
        df["k"] = k
        df["m"] = m
        df["source_file"] = f.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def summarize_group(g: pd.DataFrame, alpha: float = ALPHA) -> pd.Series:
    """Compute summary statistics for one (m, k) simulation group.

    Fit diagnostics and calibration errors are summarized with medians
    and IQRs because they can be skewed, while uniformity-test outcomes
    are reported as rejection rates rather than mean p-values.

    Args:
        g: Rows for a single (m, k) group, containing the metric columns
            produced by the simulations (e.g. "elpd", "ks_pvalue").
        alpha: Significance threshold used for the test rejection
            rates. Defaults to ALPHA.

    Returns:
        A Series of named summary statistics indexed by metric name.

    Example:
        ```python
        stats = df_all.groupby(["m", "k"]).apply(summarize_group)
        ```

    """
    return pd.Series(
        {
            "n_simulations": len(g),
            # --- fit diagnostics (median + IQR since these can be skewed) ---
            "elpd_median": g["elpd"].median(),
            "elpd_iqr": g["elpd"].quantile(0.75) - g["elpd"].quantile(0.25),
            "eff_params_median": g["model_eff_params"].median(),
            "loo_r2_median": g["loo_rsquared"].median(),
            "frac_k_above_good_k_psis_median": g["frac_k_above_good_k_psis"].median(),
            "frac_k_above_good_k_loo_exp": g["frac_k_above_good_k_loo_exp"].median(),
            # --- custom calibration check ---
            "cal_error_median": g["calibration_error"].median(),
            "cal_error_iqr": g["calibration_error"].quantile(0.75)
            - g["calibration_error"].quantile(0.25),
            "wtd_cal_error_median": g["weighted_cal_error"].median(),
            "n_miscal_median": g["n_miscalibrated"].median(),
            # --- formal uniformity tests: report REJECTION RATE, not mean p-value ---
            "ks_reject_rate": (g["ks_pvalue"] < alpha).mean(),
            "cvm_reject_rate": (g["cvm_pvalue"] < alpha).mean(),
            "az_reject_rate": (g["p_val_az"] < alpha).mean(),
            "az_cov_reject_rate": (g["p_val_coverage_az"] < alpha).mean(),
            "diff_entropy_mean": g["diff_entropy"].mean(),
            "diff_entropy_median": g["diff_entropy"].median(),
        }
    )


def build_summary(df_all: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Summarize all simulation groups into one row per (m, k).

    Each group is reduced via `summarize_group`, and the grouping keys
    are re-ordered so that "k" appears before "m" in the output.

    Args:
        df_all: Concatenated simulation results with "m" and "k" columns,
            as produced by `load_all`.
        alpha: Significance threshold passed through to
            `summarize_group`. Defaults to ALPHA.

    Returns:
        A DataFrame with one row per (k, m) pair and one column per
        summary statistic.

    Example:
        ```python
        summary = build_summary(df_all, alpha=0.05)
        ```

    """

    def _summarize(g: pd.DataFrame) -> pd.Series:
        return summarize_group(g, alpha)

    summary = (
        df_all.groupby(["m", "k"], group_keys=True).apply(_summarize).reset_index()
    )

    cols = summary.columns.tolist()
    # put k before m
    cols.remove("k")
    cols.remove("m")
    return cast(pd.DataFrame, summary[["k", "m", *cols]])


def make_pivoted_sheets(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot the summary into one k x m matrix per metric.

    Args:
        summary: Long-format summary with "k", "m", and metric columns,
            as produced by `build_summary`.

    Returns:
        A dict mapping each metric column name to its k x m pivot table,
        suitable for writing as Excel sheets.

    Example:
        ```python
        sheets = make_pivoted_sheets(summary)
        mat = sheets["elpd_median"]
        ```

    """
    metric_cols = [c for c in summary.columns if c not in {"m", "k"}]
    sheets = {}
    for col in metric_cols:
        sheets[col] = summary.pivot_table(index="k", columns="m", values=col)
    return sheets


if __name__ == "__main__":
    df_all = load_all(str(CALIB_SIM_RES_DIR), FILE_GLOB)
    print(
        f"Loaded {len(df_all)} rows across {df_all[['k', 'm']].drop_duplicates().shape[0]} (m,m) combinations"
    )

    summary = build_summary(df_all, ALPHA)
    summary["_sort"] = [
        _km_sort_key(k, m) for k, m in zip(summary["k"], summary["m"], strict=True)
    ]
    summary = (
        summary.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== Long-format summary (one row per m,k) ===")
    print(summary.round(4).to_string(index=False))

    summary.to_csv(OUT_CSV, index=False)
    print(f"\nSaved long-format table -> {OUT_CSV}")

    # pivoted view: one k x m matrix per metric, all in one Excel file (one sheet each)
    sheets = make_pivoted_sheets(summary)
    with pd.ExcelWriter(OUT_XLSX) as writer:
        summary.to_excel(writer, sheet_name="long_format", index=False)
        for name, mat in sheets.items():
            # Excel sheet names capped at 31 chars
            sheet_name = name[:31]
            mat.round(4).to_excel(writer, sheet_name=sheet_name)
    print(f"Saved pivoted per-metric matrices -> {OUT_XLSX}")
