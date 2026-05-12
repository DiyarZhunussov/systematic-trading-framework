"""
walk_forward.py — Purged Walk-Forward Validation
Implements Section 5.2 of the framework exactly.

Core principle:
    Walk-forward validation is the PRIMARY DEFENCE against temporal data leakage.
    Purging and embargo periods are NOT OPTIONAL — they are the minimum requirement
    to prevent the training window from contaminating the test window through
    overlapping observations.

Architecture (Section 5.2):
    Full data: T total periods, divided into K folds of length L
    For each fold k:
        Training window : [0, k×L − embargo] (CUMULATIVE, not rolling)
        Embargo period  : [k×L − embargo, k×L] (excluded from both)
        Test window     : [k×L, (k+1)×L]

    WHY cumulative (not rolling):
        For strategies where parameter stability is expected (trend following),
        more historical data produces more stable parameter estimates.
        Rolling windows discard potentially valuable long-term structural info.
        Use rolling ONLY when there is explicit economic justification for
        discarding older observations.

    Minimum embargo length:
        = 2 × maximum expected autocorrelation decay time
        Typical: 5–10 trading days for daily data
                 20–40 bars for 4H data (5–10 trading days equivalent)

    Acceptance criterion (Section 5.2):
        Walk-forward Sharpe / In-sample Sharpe > 0.60
        Degradation > 40% from IS to OOS → material overfitting

    3-year dataset example (756 trading days):
        K=5 folds, L=150 days, embargo=10 days
        Fold 1: Train [0–140],   Embargo [140–150],  Test [150–300]
        Fold 2: Train [0–290],   Embargo [290–310],  Test [310–460]
        Fold 3: Train [0–450],   Embargo [450–460],  Test [460–610]
        Fold 4: Train [0–600],   Embargo [600–610],  Test [610–756]
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FOLD DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WalkForwardFold:
    """Definition of a single walk-forward fold."""
    fold_index: int
    train_start: int
    train_end: int          # Exclusive — does not include embargo
    embargo_start: int
    embargo_end: int        # Exclusive
    test_start: int
    test_end: int           # Exclusive

    @property
    def train_length(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_length(self) -> int:
        return self.test_end - self.test_start

    @property
    def embargo_length(self) -> int:
        return self.embargo_end - self.embargo_start

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_index}: "
            f"Train[{self.train_start}–{self.train_end}] "
            f"Embargo[{self.embargo_start}–{self.embargo_end}] "
            f"Test[{self.test_start}–{self.test_end}]"
        )


@dataclass
class FoldResult:
    """Result of running a strategy on one fold."""
    fold: WalkForwardFold
    is_sharpe: float            # In-sample Sharpe
    oos_sharpe: float           # Out-of-sample Sharpe
    is_returns: np.ndarray
    oos_returns: np.ndarray
    best_params: Optional[dict] = None
    degradation: float = 0.0    # (IS - OOS) / IS

    def __post_init__(self):
        if abs(self.is_sharpe) > 1e-10:
            self.degradation = (self.is_sharpe - self.oos_sharpe) / abs(self.is_sharpe)


@dataclass
class WalkForwardResult:
    """Aggregated result across all folds."""
    n_folds: int
    fold_results: list[FoldResult]
    is_sharpe_mean: float           # Mean IS Sharpe across folds
    oos_sharpe_aggregate: float     # Aggregate OOS Sharpe (all OOS periods)
    oos_sharpe_mean: float          # Mean OOS Sharpe across folds
    is_to_oos_ratio: float          # oos_agg / is_mean
    passes: bool                    # Ratio > 0.60
    verdict: str
    embargo_days: int
    fold_length: int
    cumulative_training: bool

    def log_summary(self) -> str:
        return (
            f"Walk-Forward | K={self.n_folds} folds | "
            f"IS_SR={self.is_sharpe_mean:.3f} | "
            f"OOS_SR={self.oos_sharpe_aggregate:.3f} | "
            f"IS/OOS ratio={self.is_to_oos_ratio:.3f} | "
            f"{'PASS' if self.passes else 'FAIL'}: {self.verdict}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FOLD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_walk_forward_folds(
    n_observations: int,
    n_folds: int = 5,
    embargo_periods: int = 10,
    min_train_periods: int = 100,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward fold definitions with cumulative training windows.

    Parameters
    ----------
    n_observations   : Total number of observations in the dataset
    n_folds          : Number of folds (K=5 recommended)
    embargo_periods  : Periods to exclude between train and test
    min_train_periods: Minimum training periods required for fold 1

    Returns
    -------
    List of WalkForwardFold definitions

    Example (PDF Section 5.2 — 756 days, K=5, L=150, embargo=10):
        Fold 1: Train[0–140], Embargo[140–150], Test[150–300]
        Fold 2: Train[0–290], Embargo[290–310], Test[310–460]
        Fold 3: Train[0–450], Embargo[450–460], Test[460–610]
        Fold 4: Train[0–600], Embargo[600–610], Test[610–756]
    """
    fold_length = n_observations // (n_folds + 1)

    if fold_length < embargo_periods + 10:
        raise ValueError(
            f"Fold length ({fold_length}) too short relative to "
            f"embargo ({embargo_periods}). Reduce n_folds or embargo_periods."
        )

    folds = []
    for k in range(1, n_folds + 1):
        train_end = k * fold_length - embargo_periods
        embargo_start = k * fold_length - embargo_periods
        embargo_end = k * fold_length
        test_start = k * fold_length
        test_end = min((k + 1) * fold_length, n_observations)

        if train_end < min_train_periods:
            logger.warning(
                f"Fold {k}: training window ({train_end}) < "
                f"minimum ({min_train_periods}). Skipping fold."
            )
            continue

        if test_end <= test_start:
            break

        fold = WalkForwardFold(
            fold_index=k,
            train_start=0,          # Cumulative: always starts at 0
            train_end=train_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            test_start=test_start,
            test_end=test_end,
        )
        folds.append(fold)
        logger.debug(repr(fold))

    logger.info(
        f"Generated {len(folds)} walk-forward folds | "
        f"fold_length={fold_length} | embargo={embargo_periods} | "
        f"cumulative_training=True"
    )
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward(
    prices: np.ndarray,
    strategy_fn: Callable[[np.ndarray, dict], np.ndarray],
    param_grid: list[dict],
    n_folds: int = 5,
    embargo_periods: int = 10,
    optimise_metric: str = "sharpe",
    min_train_periods: int = 100,
) -> WalkForwardResult:
    """
    Run purged walk-forward validation.

    Parameters
    ----------
    prices        : Full price series (T,)
    strategy_fn   : Callable(prices_train, params) → daily_returns array
                    Must use ONLY prices_train when computing signals
    param_grid    : List of parameter dicts to search on each fold's IS window
    n_folds       : Number of folds (K)
    embargo_periods : Bars to exclude between train and test
    optimise_metric : Metric to maximise on IS window ("sharpe" | "sortino")
    min_train_periods : Minimum IS observations for first fold

    Returns
    -------
    WalkForwardResult with full fold-level diagnostics

    Usage:
        def my_strategy(prices, params):
            z = rolling_zscore(prices, params['window'])
            positions = np.where(z < -params['threshold'], 1,
                        np.where(z > params['threshold'], -1, 0))
            returns = np.diff(prices) / prices[:-1]
            return positions[:-1] * returns

        result = run_walk_forward(
            prices=my_prices,
            strategy_fn=my_strategy,
            param_grid=[{'window': w, 'threshold': t}
                        for w in [20,30,40] for t in [1.5, 2.0, 2.5]],
            n_folds=5,
            embargo_periods=10,
        )
    """
    n = len(prices)
    folds = generate_walk_forward_folds(n, n_folds, embargo_periods, min_train_periods)

    if not folds:
        raise ValueError("No valid folds generated — dataset too short")

    fold_results = []
    all_oos_returns = []

    for fold in folds:
        # ── IS: find best parameters ──────────────────────────────────────────
        train_prices = prices[fold.train_start:fold.train_end]

        best_params = None
        best_metric = -np.inf

        for params in param_grid:
            try:
                returns = strategy_fn(train_prices, params)
                if len(returns) < 10:
                    continue
                metric = _compute_metric(returns, optimise_metric)
                if metric > best_metric:
                    best_metric = metric
                    best_params = params
            except Exception as e:
                logger.debug(f"Fold {fold.fold_index} param search error: {e}")
                continue

        if best_params is None:
            logger.warning(f"Fold {fold.fold_index}: no valid params found — skipping")
            continue

        is_returns = strategy_fn(train_prices, best_params)
        is_sharpe = _compute_metric(is_returns, "sharpe")

        # ── OOS: evaluate best params on held-out test window ─────────────────
        # CRITICAL: test window uses prices from test_start, but strategy
        # receives only the test prices (no lookahead into training period)
        test_prices = prices[fold.test_start:fold.test_end]
        try:
            oos_returns = strategy_fn(test_prices, best_params)
        except Exception as e:
            logger.warning(f"Fold {fold.fold_index} OOS evaluation failed: {e}")
            continue

        oos_sharpe = _compute_metric(oos_returns, "sharpe")
        all_oos_returns.extend(oos_returns.tolist())

        fr = FoldResult(
            fold=fold,
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            is_returns=is_returns,
            oos_returns=oos_returns,
            best_params=best_params,
        )
        fold_results.append(fr)
        logger.info(
            f"Fold {fold.fold_index}: IS_SR={is_sharpe:.3f} | "
            f"OOS_SR={oos_sharpe:.3f} | "
            f"degradation={fr.degradation:.1%} | "
            f"best_params={best_params}"
        )

    if not fold_results:
        raise ValueError("All folds failed — cannot compute walk-forward result")

    # ── Aggregate results ──────────────────────────────────────────────────────
    is_sharpes = [fr.is_sharpe for fr in fold_results]
    is_sharpe_mean = float(np.mean(is_sharpes))

    all_oos = np.array(all_oos_returns)
    oos_sharpe_agg = _compute_metric(all_oos, "sharpe")
    oos_sharpe_mean = float(np.mean([fr.oos_sharpe for fr in fold_results]))

    is_to_oos_ratio = (
        oos_sharpe_agg / abs(is_sharpe_mean)
        if abs(is_sharpe_mean) > 1e-10 else 0.0
    )
    passes = is_to_oos_ratio >= 0.60

    if is_to_oos_ratio < 0.50:
        verdict = "STRONG_OVERFITTING — reject strategy"
    elif is_to_oos_ratio < 0.60:
        verdict = "MARGINAL — simplify model, re-test"
    elif is_to_oos_ratio < 0.80:
        verdict = "ACCEPTABLE — proceed to Stage 3 validation"
    else:
        verdict = "GOOD — solid OOS performance"

    result = WalkForwardResult(
        n_folds=len(fold_results),
        fold_results=fold_results,
        is_sharpe_mean=is_sharpe_mean,
        oos_sharpe_aggregate=oos_sharpe_agg,
        oos_sharpe_mean=oos_sharpe_mean,
        is_to_oos_ratio=is_to_oos_ratio,
        passes=passes,
        verdict=verdict,
        embargo_days=embargo_periods,
        fold_length=n // (n_folds + 1),
        cumulative_training=True,
    )

    logger.info(f"Walk-forward complete: {result.log_summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HOLD-OUT VALIDATOR
# Final validation on the never-touch set (Section 5.6)
# ─────────────────────────────────────────────────────────────────────────────
def final_holdout_validation(
    prices: np.ndarray,
    strategy_fn: Callable[[np.ndarray, dict], np.ndarray],
    best_params: dict,
    holdout_fraction: float = 0.20,
) -> dict:
    """
    Validate strategy on the 20% hold-out set.

    CRITICAL RULE (Section 5.6):
        This data is viewed ONCE — after ALL parameter decisions are finalised.
        If OOS hold-out Sharpe < 50% of IS Sharpe: STRATEGY IS REJECTED.
        No exceptions. No re-testing on this dataset.

    Parameters
    ----------
    prices           : Full price series
    strategy_fn      : Strategy function(prices, params) → returns
    best_params      : Final parameters (must be locked before calling this)
    holdout_fraction : Fraction of data reserved as hold-out (default 20%)

    Returns
    -------
    dict with: is_sharpe, holdout_sharpe, ratio, passes, verdict
    """
    n = len(prices)
    holdout_start = int(n * (1 - holdout_fraction))

    if holdout_start < 50:
        raise ValueError(
            f"Hold-out set too small ({n - holdout_start} obs). "
            f"Need at least 50 observations."
        )

    is_prices = prices[:holdout_start]
    holdout_prices = prices[holdout_start:]

    is_returns = strategy_fn(is_prices, best_params)
    holdout_returns = strategy_fn(holdout_prices, best_params)

    is_sharpe = _compute_metric(is_returns, "sharpe")
    holdout_sharpe = _compute_metric(holdout_returns, "sharpe")

    ratio = (
        holdout_sharpe / abs(is_sharpe)
        if abs(is_sharpe) > 1e-10 else 0.0
    )
    passes = ratio >= 0.50

    verdict = (
        "PASS — proceed to paper trading"
        if passes
        else "FAIL — strategy rejected (holdout SR < 50% of IS SR)"
    )

    logger.info(
        f"Hold-out validation: IS_SR={is_sharpe:.3f} | "
        f"holdout_SR={holdout_sharpe:.3f} | ratio={ratio:.3f} | "
        f"{'PASS' if passes else 'FAIL'}"
    )

    return {
        "is_sharpe": float(is_sharpe),
        "holdout_sharpe": float(holdout_sharpe),
        "ratio": float(ratio),
        "passes": passes,
        "verdict": verdict,
        "holdout_n_obs": len(holdout_prices),
        "is_n_obs": len(is_prices),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _compute_metric(returns: np.ndarray, metric: str) -> float:
    """Compute optimisation metric from return series."""
    returns = np.asarray(returns, dtype=float)
    valid = returns[~np.isnan(returns)]
    if len(valid) < 5:
        return -999.0
    mean_r = float(np.mean(valid))
    std_r = float(np.std(valid, ddof=1))
    if std_r < 1e-10:
        return 0.0

    if metric == "sharpe":
        return float(mean_r / std_r * np.sqrt(252))
    elif metric == "sortino":
        downside = valid[valid < 0]
        if len(downside) < 2:
            return float(mean_r / std_r * np.sqrt(252))
        dd_std = float(np.std(downside, ddof=1))
        return float(mean_r / (dd_std + 1e-10) * np.sqrt(252))
    else:
        return float(mean_r / std_r * np.sqrt(252))
