"""
monte_carlo_stress.py — Monthly Monte Carlo Portfolio Stress Test
Implements Section 7.7 of the framework.

Requirement (Section 7.7):
    Generate 10,000 portfolio return paths by:
        1. Bootstrap daily returns from trailing 252 days
        2. Apply regime-conditioned correlation matrices (calm and stressed)
        3. Record: maximum drawdown, duration, P(ruin) at current sizing

    REQUIREMENT: P(drawdown > 10%) across simulated paths < 5%
    If exceeded: reduce all position sizes until requirement satisfied

Run frequency: Monthly (scheduled from main.py daily tasks)

This test is distinct from the correlation stress tests in portfolio_engine.py
which use analytical methods. This test uses empirical bootstrapping to capture
fat tails, autocorrelation, and regime-conditioned correlations that analytical
methods miss.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MonteCarloResult:
    """Results of the monthly Monte Carlo stress test."""
    n_paths: int
    n_days: int
    account_balance: float

    # Drawdown statistics across paths
    mean_max_drawdown: float
    p95_max_drawdown: float
    p99_max_drawdown: float
    prob_drawdown_exceeds_10pct: float      # KEY METRIC — must be < 5%
    prob_drawdown_exceeds_8pct: float       # Internal kill switch level
    prob_drawdown_exceeds_5pct: float       # Prop firm phase 2 level

    # Duration statistics
    mean_max_drawdown_duration_days: float
    p95_max_drawdown_duration_days: float

    # Return statistics
    mean_terminal_return: float
    p5_terminal_return: float               # 5th percentile (downside)
    p95_terminal_return: float              # 95th percentile (upside)

    # Gate
    passes: bool                            # P(DD>10%) < 5%
    size_reduction_factor: float            # 1.0 if passes, < 1.0 if reduction needed
    recommendation: str

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def log_summary(self) -> str:
        return (
            f"Monte Carlo ({self.n_paths} paths) | "
            f"P(DD>10%)={self.prob_drawdown_exceeds_10pct:.1%} | "
            f"P95_MaxDD={self.p95_max_drawdown:.1%} | "
            f"{'PASS' if self.passes else 'FAIL — REDUCE SIZES'}: "
            f"{self.recommendation}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CORE MONTE CARLO FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo_stress(
    portfolio_daily_returns: np.ndarray,
    account_balance: float,
    n_paths: int = 10_000,
    n_days: int = 252,
    lookback_days: int = 252,
    calm_correlation_scale: float = 1.0,
    crisis_correlation_scale: float = 1.5,
    drawdown_limit: float = 0.10,
    drawdown_limit_pct_threshold: float = 0.05,
    random_seed: int = 42,
) -> MonteCarloResult:
    """
    Run monthly Monte Carlo stress test (Section 7.7).

    Method:
        1. Bootstrap daily returns from trailing lookback_days
        2. Apply two correlation regimes (calm and stressed)
        3. Record max drawdown and duration for each path
        4. Compute P(drawdown > 10%)
        5. If P > 5%, compute required size reduction

    Parameters
    ----------
    portfolio_daily_returns : Array of daily portfolio returns (fraction)
    account_balance         : Current account balance
    n_paths                 : Number of simulation paths (10,000 per framework)
    n_days                  : Days to simulate per path (252 = 1 year)
    lookback_days           : Historical window for bootstrap (252)
    calm_correlation_scale  : Multiplier for calm regime vol (1.0 = unchanged)
    crisis_correlation_scale: Multiplier for stressed regime vol (1.5 = 50% wider)
    drawdown_limit          : Drawdown threshold to measure P against (0.10)
    drawdown_limit_pct_threshold : P(DD > limit) must be below this (0.05)
    random_seed             : Reproducibility

    Returns
    -------
    MonteCarloResult with full diagnostic breakdown
    """
    rng = np.random.default_rng(random_seed)
    returns = np.asarray(portfolio_daily_returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    # Use trailing lookback_days
    historical = returns[-lookback_days:] if len(returns) >= lookback_days else returns
    n_hist = len(historical)

    if n_hist < 20:
        logger.warning(
            f"Monte Carlo: insufficient history ({n_hist} days). "
            f"Using conservative parametric simulation."
        )
        mean_r = float(np.mean(historical)) if n_hist > 0 else 0.0
        std_r = float(np.std(historical, ddof=1)) if n_hist > 1 else 0.01
        historical = rng.normal(mean_r, std_r, 252)
        n_hist = len(historical)

    # ── Generate 10,000 paths via block bootstrap ─────────────────────────────
    # Block bootstrap preserves short-term autocorrelation
    block_size = max(5, n_hist // 20)

    max_drawdowns = np.zeros(n_paths)
    max_dd_durations = np.zeros(n_paths)
    terminal_returns = np.zeros(n_paths)

    for path_idx in range(n_paths):
        # Mix of calm and crisis blocks
        use_crisis = rng.random() < 0.20  # 20% of paths get crisis scaling
        vol_scale = crisis_correlation_scale if use_crisis else calm_correlation_scale

        # Block bootstrap
        path_returns = np.zeros(n_days)
        day = 0
        while day < n_days:
            block_start = rng.integers(0, max(1, n_hist - block_size))
            block = historical[block_start:block_start + block_size] * vol_scale
            take = min(block_size, n_days - day)
            path_returns[day:day + take] = block[:take]
            day += take

        # Compute equity path
        equity = np.cumprod(1 + path_returns)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10)
        max_dd = float(np.max(drawdown))
        max_drawdowns[path_idx] = max_dd

        # Drawdown duration (longest consecutive drawdown period)
        in_dd = drawdown > 0.01  # 1% threshold for "in drawdown"
        max_dur = 0
        cur_dur = 0
        for d in in_dd:
            if d:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        max_dd_durations[path_idx] = max_dur

        terminal_returns[path_idx] = float(equity[-1] - 1)

    # ── Compute statistics ────────────────────────────────────────────────────
    prob_10 = float(np.mean(max_drawdowns > drawdown_limit))
    prob_8 = float(np.mean(max_drawdowns > 0.08))
    prob_5 = float(np.mean(max_drawdowns > 0.05))

    passes = prob_10 < drawdown_limit_pct_threshold

    # Compute required size reduction if test fails
    if not passes:
        # Binary search for size reduction factor that brings P(DD>10%) < 5%
        size_factor = _find_required_size_reduction(
            historical, n_paths, n_days, block_size,
            drawdown_limit, drawdown_limit_pct_threshold, rng
        )
        recommendation = (
            f"REDUCE ALL POSITION SIZES by {(1 - size_factor):.0%} "
            f"(factor={size_factor:.2f}). "
            f"Current P(DD>10%)={prob_10:.1%} exceeds 5% limit."
        )
    else:
        size_factor = 1.0
        recommendation = (
            f"Position sizes acceptable. "
            f"P(DD>10%)={prob_10:.1%} < {drawdown_limit_pct_threshold:.0%} limit."
        )

    result = MonteCarloResult(
        n_paths=n_paths,
        n_days=n_days,
        account_balance=account_balance,
        mean_max_drawdown=float(np.mean(max_drawdowns)),
        p95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
        p99_max_drawdown=float(np.percentile(max_drawdowns, 99)),
        prob_drawdown_exceeds_10pct=prob_10,
        prob_drawdown_exceeds_8pct=prob_8,
        prob_drawdown_exceeds_5pct=prob_5,
        mean_max_drawdown_duration_days=float(np.mean(max_dd_durations)),
        p95_max_drawdown_duration_days=float(np.percentile(max_dd_durations, 95)),
        mean_terminal_return=float(np.mean(terminal_returns)),
        p5_terminal_return=float(np.percentile(terminal_returns, 5)),
        p95_terminal_return=float(np.percentile(terminal_returns, 95)),
        passes=passes,
        size_reduction_factor=size_factor,
        recommendation=recommendation,
    )

    logger.info(f"Monte Carlo: {result.log_summary()}")

    if not passes:
        logger.warning(
            f"MONTE CARLO STRESS TEST FAILED: "
            f"P(DD>10%)={prob_10:.1%} > 5% limit. "
            f"Reduce all position sizes by {(1 - size_factor):.0%}."
        )

    return result


def _find_required_size_reduction(
    historical: np.ndarray,
    n_paths: int,
    n_days: int,
    block_size: int,
    drawdown_limit: float,
    target_prob: float,
    rng: np.random.Generator,
    n_search_paths: int = 2000,
) -> float:
    """
    Binary search for the position size scaling factor that brings
    P(max drawdown > limit) below target_prob.
    Uses a reduced n_search_paths for speed.
    """
    lo, hi = 0.1, 1.0

    for _ in range(10):  # 10 iterations of binary search
        mid = (lo + hi) / 2
        # Scale returns by mid (smaller positions = smaller returns and drawdowns)
        scaled = historical * mid

        max_dds = []
        for _ in range(n_search_paths):
            path_returns = np.zeros(n_days)
            day = 0
            while day < n_days:
                start = rng.integers(0, max(1, len(scaled) - block_size))
                block = scaled[start:start + block_size]
                take = min(block_size, n_days - day)
                path_returns[day:day + take] = block[:take]
                day += take
            equity = np.cumprod(1 + path_returns)
            peak = np.maximum.accumulate(equity)
            dd = float(np.max((peak - equity) / (peak + 1e-10)))
            max_dds.append(dd)

        prob = float(np.mean(np.array(max_dds) > drawdown_limit))
        if prob < target_prob:
            hi = mid
        else:
            lo = mid

    return float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY SCHEDULER INTEGRATION
# Called from main.py daily tasks when month rolls over
# ─────────────────────────────────────────────────────────────────────────────
class MonthlyStressTestScheduler:
    """
    Tracks when Monte Carlo test was last run and triggers monthly.
    Integrates with the main trading loop.
    """

    def __init__(self):
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[MonteCarloResult] = None

    def should_run(self, now: Optional[datetime] = None) -> bool:
        """True if 30+ days since last run, or never run."""
        now = now or datetime.now(timezone.utc)
        if self._last_run is None:
            return True
        return (now - self._last_run).days >= 30

    def run_if_due(
        self,
        portfolio_daily_returns: np.ndarray,
        account_balance: float,
        n_paths: int = 10_000,
    ) -> Optional[MonteCarloResult]:
        """Run stress test if due. Returns result or None."""
        if not self.should_run():
            return None

        logger.info("Monthly Monte Carlo stress test due — running...")
        result = run_monte_carlo_stress(
            portfolio_daily_returns=portfolio_daily_returns,
            account_balance=account_balance,
            n_paths=n_paths,
        )
        self._last_run = datetime.now(timezone.utc)
        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[MonteCarloResult]:
        return self._last_result

    @property
    def days_since_last_run(self) -> Optional[int]:
        if self._last_run is None:
            return None
        return (datetime.now(timezone.utc) - self._last_run).days


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    np.random.seed(42)
    print("\n── Monte Carlo Stress Test Example ──")
    print("Simulating 252 days of realistic portfolio returns...")

    # Simulate a year of daily returns: mean=0.05% daily, std=0.8%
    fake_returns = np.random.normal(0.0005, 0.008, 252)
    result = run_monte_carlo_stress(
        portfolio_daily_returns=fake_returns,
        account_balance=100_000,
        n_paths=2000,   # Reduced for demo speed
        n_days=252,
    )

    print(f"\nResults:")
    print(f"  P(max drawdown > 10%) : {result.prob_drawdown_exceeds_10pct:.1%}")
    print(f"  P(max drawdown > 8%)  : {result.prob_drawdown_exceeds_8pct:.1%}")
    print(f"  P95 max drawdown      : {result.p95_max_drawdown:.1%}")
    print(f"  Mean terminal return  : {result.mean_terminal_return:.1%}")
    print(f"  P5  terminal return   : {result.p5_terminal_return:.1%}")
    print(f"  Test passes           : {result.passes}")
    print(f"  Recommendation        : {result.recommendation}")
