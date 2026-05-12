"""
pbo.py — Probability of Backtest Overfitting (PBO)
Implements Section 5.4 of the framework.

Reference: Bailey et al. (2016) — Combinatorially Symmetric Cross-Validation (CSCV)

The PBO provides a direct estimate of the probability that the best
in-sample strategy configuration is a product of noise rather than
genuine edge.

Procedure (Section 5.4):
    1. Divide the backtest period into S subsamples (S=16 recommended)
    2. Generate C(S, S/2) = 12,870 combinations of training/testing splits
    3. For each combination:
        a. Rank strategies by Sharpe on training subset → select optimal
        b. Measure Sharpe of that same configuration on the testing subset
    4. PBO = fraction of combinations where the IS-optimal strategy
             underperforms the MEDIAN strategy in the OOS test set

Interpretation (Section 5.4):
    PBO < 0.10  : Low overfitting — acceptable for further validation
    PBO 0.10–0.25: Moderate — simplify model before proceeding
    PBO > 0.25  : High overfitting — reject or substantially redesign
    PBO > 0.50  : Strategy is predominantly noise — retire hypothesis

Note on computational cost:
    C(16,8) = 12,870 combinations. For large strategy universes (N>50),
    this can be slow. Use n_combinations to cap at a random subsample.
    Results are still reliable with n_combinations=2000.
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PBOResult:
    """Full output of PBO computation."""
    pbo: float                          # Probability of backtest overfitting
    n_combinations: int                 # Number of IS/OOS splits evaluated
    n_subsamples: int                   # S
    n_strategies: int                   # Number of strategy variants
    oos_sharpes: list[float]            # OOS Sharpe for IS-optimal per combination
    median_oos_sharpes: list[float]     # Median OOS Sharpe across strategies per combo
    logit_pbo: float                    # Logit-transformed PBO (more informative)
    verdict: str
    passes: bool

    def log_summary(self) -> str:
        return (
            f"PBO={self.pbo:.4f} | "
            f"logit={self.logit_pbo:.3f} | "
            f"n_combos={self.n_combinations} | "
            f"{'PASS' if self.passes else 'FAIL'}: {self.verdict}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PBO FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def probability_of_backtest_overfitting(
    strategy_returns_matrix: np.ndarray,
    n_subsamples: int = 16,
    n_combinations: Optional[int] = None,
    random_seed: int = 42,
) -> PBOResult:
    """
    Compute Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    strategy_returns_matrix : (T × N) array
                              T = time periods, N = strategy variants
                              Each column is the return series of one variant.
    n_subsamples            : S — number of subsamples (16 recommended)
    n_combinations          : Cap on number of IS/OOS splits (None = all C(S,S/2))
                              Use 2000 for large N to save compute.
    random_seed             : Random seed for combination sampling

    Returns
    -------
    PBOResult with PBO estimate and diagnostics
    """
    T, N = strategy_returns_matrix.shape

    if N < 2:
        raise ValueError(f"Need at least 2 strategy variants for PBO (got {N})")
    if T < n_subsamples * 5:
        raise ValueError(
            f"Too few observations ({T}) for {n_subsamples} subsamples. "
            f"Need at least {n_subsamples * 5}."
        )

    # ── Step 1: Divide into S subsamples ─────────────────────────────────────
    subsample_size = T // n_subsamples
    subsamples = []
    for s in range(n_subsamples):
        start = s * subsample_size
        end = start + subsample_size
        subsamples.append(strategy_returns_matrix[start:end, :])

    # ── Step 2: Generate C(S, S/2) combinations ───────────────────────────────
    half_s = n_subsamples // 2
    all_indices = list(range(n_subsamples))
    all_combos = list(combinations(all_indices, half_s))

    # Optionally cap at n_combinations
    if n_combinations is not None and len(all_combos) > n_combinations:
        rng = np.random.default_rng(random_seed)
        selected_idx = rng.choice(len(all_combos), size=n_combinations, replace=False)
        all_combos = [all_combos[i] for i in selected_idx]

    logger.info(
        f"PBO: T={T}, N={N}, S={n_subsamples}, "
        f"evaluating {len(all_combos)} combinations"
    )

    # ── Step 3: Evaluate each combination ────────────────────────────────────
    oos_sharpes_of_optimal = []
    median_oos_sharpes = []
    n_underperform = 0

    for combo in all_combos:
        is_indices = list(combo)
        oos_indices = [i for i in all_indices if i not in is_indices]

        # Stack IS and OOS subsamples
        is_returns = np.vstack([subsamples[i] for i in is_indices])   # (T/2 × N)
        oos_returns = np.vstack([subsamples[i] for i in oos_indices]) # (T/2 × N)

        # Compute Sharpe for each strategy on IS window
        is_sharpes = _sharpe_per_strategy(is_returns)

        # Select IS-optimal strategy
        best_is_idx = int(np.argmax(is_sharpes))

        # Measure OOS Sharpe of all strategies
        oos_sharpes = _sharpe_per_strategy(oos_returns)

        oos_of_optimal = float(oos_sharpes[best_is_idx])
        median_oos = float(np.median(oos_sharpes))

        oos_sharpes_of_optimal.append(oos_of_optimal)
        median_oos_sharpes.append(median_oos)

        # Count: did IS-optimal underperform median in OOS?
        if oos_of_optimal < median_oos:
            n_underperform += 1

    # ── Step 4: PBO = fraction where IS-optimal underperforms OOS median ─────
    n_total = len(all_combos)
    pbo = n_underperform / n_total if n_total > 0 else 0.5

    # Logit transformation for interpretability (0.5 = random, <0 = good, >0 = bad)
    logit_pbo = float(np.log(pbo / (1 - pbo + 1e-10) + 1e-10))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if pbo < 0.10:
        verdict = "LOW_OVERFITTING — proceed to further validation"
        passes = True
    elif pbo < 0.25:
        verdict = "MODERATE_OVERFITTING — simplify model before proceeding"
        passes = False
    elif pbo < 0.50:
        verdict = "HIGH_OVERFITTING — reject or substantially redesign"
        passes = False
    else:
        verdict = "PREDOMINANTLY_NOISE — retire hypothesis"
        passes = False

    result = PBOResult(
        pbo=float(pbo),
        n_combinations=n_total,
        n_subsamples=n_subsamples,
        n_strategies=N,
        oos_sharpes=oos_sharpes_of_optimal,
        median_oos_sharpes=median_oos_sharpes,
        logit_pbo=logit_pbo,
        verdict=verdict,
        passes=passes,
    )

    logger.info(f"PBO: {result.log_summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GRID BUILDER
# Converts a strategy with a price series into a returns matrix for PBO
# ─────────────────────────────────────────────────────────────────────────────
def build_strategy_returns_matrix(
    prices: np.ndarray,
    strategy_fn,
    param_grid: list[dict],
) -> np.ndarray:
    """
    Build the (T × N) strategy returns matrix for PBO input.

    Parameters
    ----------
    prices      : Price series
    strategy_fn : Callable(prices, params) → returns array of length T-1
    param_grid  : List of parameter dicts — one column per dict

    Returns
    -------
    (T-1 × N) returns matrix (NaN-filled for strategies with insufficient data)
    """
    T = len(prices) - 1
    N = len(param_grid)
    matrix = np.full((T, N), np.nan)

    for j, params in enumerate(param_grid):
        try:
            returns = strategy_fn(prices, params)
            n = min(len(returns), T)
            matrix[T - n:, j] = returns[-n:]
        except Exception as e:
            logger.debug(f"Param {j} ({params}) failed: {e}")

    # Remove rows that are all NaN (early rows before all strategies have data)
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows]

    # Replace remaining NaNs with 0 (strategy not active)
    matrix = np.nan_to_num(matrix, nan=0.0)

    logger.info(
        f"Built strategy returns matrix: "
        f"{matrix.shape[0]} periods × {N} strategies"
    )
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sharpe_per_strategy(returns_matrix: np.ndarray) -> np.ndarray:
    """Compute annualised Sharpe for each column of returns_matrix."""
    N = returns_matrix.shape[1]
    sharpes = np.zeros(N)
    for j in range(N):
        col = returns_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 5:
            sharpes[j] = -999.0
            continue
        std = float(np.std(valid, ddof=1))
        if std < 1e-10:
            sharpes[j] = 0.0
        else:
            sharpes[j] = float(np.mean(valid) / std * np.sqrt(252))
    return sharpes


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    np.random.seed(42)
    print("\n── PBO Example: 20 strategies, 500 time periods ──")

    # Simulate 20 strategy variants — all pure noise
    noise_matrix = np.random.normal(0, 0.01, (500, 20))
    result = probability_of_backtest_overfitting(
        noise_matrix, n_subsamples=16, n_combinations=500
    )
    print(f"Pure noise PBO: {result.pbo:.3f} (should be ~0.50)")
    print(f"Verdict: {result.verdict}")

    print("\n── PBO Example: 20 strategies, 1 has genuine edge ──")
    # One strategy with a small genuine edge
    edge_matrix = np.random.normal(0, 0.01, (500, 20))
    edge_matrix[:, 0] = np.random.normal(0.0004, 0.01, 500)  # Small edge
    result2 = probability_of_backtest_overfitting(
        edge_matrix, n_subsamples=16, n_combinations=500
    )
    print(f"With edge PBO: {result2.pbo:.3f} (should be < 0.50)")
    print(f"Verdict: {result2.verdict}")
    print(f"Passes: {result2.passes}")
