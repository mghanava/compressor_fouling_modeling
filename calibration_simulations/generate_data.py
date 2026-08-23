"""Data generator for the Type I error / LOO-PIT simulation study.

Data-generating process (DGP), per the paper (LOO-PIT predictive model checking
@ https://arxiv.org/abs/2603.02928):

    y_ig | x_ig, mu_g ~ N(f(x_ig) + mu_g, sigma^2)
    mu_g ~ N(0, 0.2^2)                      (group-level random effect)
    x_ig ~ U(-2, 2)                          (covariate)
    f(x) = 0.6*sin(pi*x) - 0.3*x^2 + 0.1*x^3 (true smooth nonlinear effect)

Fixed across all simulation cells: G = 50 groups.
Varied in the simulation grid (Table 1): m (obs/group) in {5, 10, 15}.
sigma is NOT specified in the paper excerpt -> exposed here as a free
parameter (default 0.3); override it to match whatever value you find
elsewhere in the paper, or to explore sensitivity.

Model complexity (k, the spline basis dimension: 6/12/24/48) does NOT
enter the DGP at all -- it only affects the *fitted* model later. Keep
that fitting code separate from this generator.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DGPParams:
    G: int = 50  # number of groups (fixed in the paper)
    m: int = 10  # observations per group (varied: 5, 10, 15)
    sigma: float = 0.3  # residual SD (not specified in paper -> free param)
    mu_sd: float = 0.2  # SD of group-level random effect mu_g
    x_low: float = -2.0
    x_high: float = 2.0


def true_f(x: np.ndarray) -> np.ndarray:
    """Simulate true smooth nonlinear covariate effect f(x).

    Args:
        x: Covariate values.

    Returns:
        True f(x) effect evaluated at x.

    Example:
        ```python
        x = np.linspace(-2.0, 2.0, 100)
        f = true_f(x)
        ```

    """
    return 0.6 * np.sin(np.pi * x) - 0.3 * x**2 + 0.1 * x**3


def generate_dataset(params: DGPParams, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a single dataset from the DGP.

    Returns a long-format DataFrame with one row per observation;
    ground-truth columns are kept for diagnostics and should be dropped
    before fitting the "unknown truth" model.

    Args:
        params: DGP parameters (group count, observations per group,
            residual SD, random-effect SD, covariate range).
        rng: Seeded NumPy generator for reproducible sampling.

    Returns:
        A long-format DataFrame with one row per observation, containing
        columns: group (group index 0..G-1), x (covariate), y (response),
        mu_g (true group-level random effect, diagnostics only), and
        f_true (true f(x), diagnostics only).

    Example:
        ```python
        params = DGPParams(G=50, m=10, sigma=0.3)
        rng = np.random.default_rng(42)
        df = generate_dataset(params, rng)
        ```

    """
    G, m = params.G, params.m

    mu_g = rng.normal(loc=0.0, scale=params.mu_sd, size=G)  # (G,)
    group_idx = np.repeat(np.arange(G), m)  # (G*m,)
    x = rng.uniform(params.x_low, params.x_high, size=G * m)  # (G*m,)

    f_true = true_f(x)
    mean = f_true + mu_g[group_idx]
    y = rng.normal(loc=mean, scale=params.sigma, size=G * m)

    return pd.DataFrame(
        {
            "group": group_idx,
            "x": x,
            "y": y,
            "mu_g": mu_g[group_idx],
            "f_true": f_true,
        }
    )


def generate_many(params: DGPParams, n_datasets: int, seed: int = 0):
    """Generate `n_datasets` independent datasets for one simulation cell.

    k (spline basis dimension) does not affect generation, so this only
    needs to be re-run once per distinct m (and sigma).

    Args:
        params: DGP parameters shared by all generated datasets.
        n_datasets: Number of independent datasets to generate
            (e.g. 5000 trials for one (k, m) cell of Table 1).
        seed: Seed for the random number generator. Defaults to 0.

    Yields:
        (trial_index, DataFrame) pairs, so 5000 datasets aren't all held
        in memory at once.

    Example:
        ```python
        params = DGPParams(G=50, m=5, sigma=0.3)
        for s, df in generate_many(params, n_datasets=30, seed=1):
            print(s, df.shape)
        ```

    """
    rng = np.random.default_rng(seed)
    for s in range(n_datasets):
        yield s, generate_dataset(params, rng)


if __name__ == "__main__":
    # Quick smoke test / example usage matching one row of Table 1: m=10
    params = DGPParams(G=50, m=10, sigma=0.3)
    rng = np.random.default_rng(42)

    df = generate_dataset(params, rng)
    print(df.head())
    print(f"\nShape: {df.shape}  (expect G*m = {params.G * params.m} rows)")
    print(f"Groups: {df['group'].nunique()} (expect {params.G})")
    print(f"x range: [{df['x'].min():.3f}, {df['x'].max():.3f}]")
    # Example: iterate a small number of trials for one grid cell
    n_trials_demo = 30
    print(f"\nGenerating {n_trials_demo} example trials for m=5, sigma=0.3:")
    params_demo = DGPParams(G=50, m=5, sigma=0.3)
    for s, d in generate_many(params_demo, n_trials_demo, seed=1):
        print(
            f"  trial {s}: {d.shape[0]} rows, y mean={d['y'].mean():.3f}, y sd={d['y'].std():.3f}"
        )
