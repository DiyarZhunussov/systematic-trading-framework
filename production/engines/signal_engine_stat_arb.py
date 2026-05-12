"""
signal_engine_stat_arb.py — Alpha Engine 5: Statistical Arbitrage
Implements Section 3, Alpha Engine 5 of the framework.

EPISTEMIC STATUS: EXPERIMENTAL — ZERO CAPITAL ALLOCATION AT LAUNCH

This engine receives NO capital allocation until:
    1. 12+ months of validated paper trading explicitly accounting for leg execution risk
    2. Either: (a) broker offers simultaneous basket/OCO execution, OR
               (b) leg risk is modelled as an irremovable structural cost from day one

The leg execution problem (Section 3, Engine 5):
    MT5 standard market order execution: 50–200ms per leg.
    Two-leg trade: combined gap of 100–400ms during which spread can move 1–3 pips.
    At a typical signal threshold of ±2σ with expected gross edge 10–15 pips:
    → 3-pip adverse leg gap = 20–30% of expected gross edge, BEFORE other costs.
    This is a NEAR-FATAL implementation barrier at retail execution speeds.

Research use only:
    This engine can generate signals for paper trading monitoring and research.
    It will NOT interact with the risk engine or order manager until allocation > 0.
    Researchers must explicitly model leg execution risk in all backtests.

Applicable pairs:
    EUR/USD / GBP/USD  — common USD driver
    NQ100 / SPX500     — correlated indices, different tech weights
    XAU/USD / USD/JPY  — both safe havens, historically negative correlation

Cointegration:
    Engle-Granger test on residuals (ADF).
    p < 0.01 required (NOT 0.05 — higher bar for this engine).
    Rolling cointegration is unstable — re-test continuously.
    If cointegration lost: IMMEDIATELY suspend the pair.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION GUARD — enforced at module level
# ─────────────────────────────────────────────────────────────────────────────
STAT_ARB_ALLOCATION_PCT = 0.0  # Must remain 0.0 until prerequisites met

def _check_allocation_guard() -> None:
    """Raise if anyone attempts to allocate capital to stat arb."""
    if STAT_ARB_ALLOCATION_PCT > 0:
        raise RuntimeError(
            "Statistical arbitrage allocation > 0 is forbidden until leg execution "
            "limitations are resolved and 12 months of validated paper trading "
            "is complete. See Section 3, Engine 5."
        )


# ─────────────────────────────────────────────────────────────────────────────
# APPROVED PAIRS
# ─────────────────────────────────────────────────────────────────────────────
APPROVED_PAIRS = [
    ("EURUSD", "GBPUSD"),    # Common USD driver
    ("NQ100", "SPX500"),     # Correlated indices
    ("XAUUSD", "USDJPY"),   # Safe haven pair (negative correlation expected)
]


class SignalDirection(Enum):
    LONG_LEG1_SHORT_LEG2 = 1
    SHORT_LEG1_LONG_LEG2 = -1
    FLAT = 0


# ─────────────────────────────────────────────────────────────────────────────
# COINTEGRATION TEST (Engle-Granger)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CointegrationResult:
    """Result of Engle-Granger cointegration test for a pair."""
    leg1: str
    leg2: str
    cointegrated: bool
    adf_statistic: float
    p_value: float
    beta: float          # Hedge ratio: Y = alpha + beta * X + epsilon
    alpha: float         # Intercept
    lookback_days: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def engle_granger_cointegration(
    prices_y: np.ndarray,
    prices_x: np.ndarray,
    leg1: str,
    leg2: str,
    p_threshold: float = 0.01,  # Higher bar than 0.05 — Engine 5 requirement
) -> CointegrationResult:
    """
    Engle-Granger cointegration test.
    Step 1: OLS regression Y = alpha + beta * X + epsilon
    Step 2: ADF test on residuals (epsilon)
    Reject non-stationarity at p < 0.01.

    Parameters
    ----------
    prices_y, prices_x : Price series for the two legs
    leg1, leg2 : Symbol strings
    p_threshold : Significance threshold (0.01 — stricter than convention)

    Returns
    -------
    CointegrationResult
    """
    n = min(len(prices_y), len(prices_x))
    if n < 60:
        return CointegrationResult(
            leg1=leg1, leg2=leg2, cointegrated=False,
            adf_statistic=np.nan, p_value=1.0,
            beta=0.0, alpha=0.0, lookback_days=n
        )

    y = prices_y[-n:]
    x = prices_x[-n:]

    # OLS: Y = alpha + beta * X
    x_with_const = np.column_stack([np.ones(n), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        return CointegrationResult(
            leg1=leg1, leg2=leg2, cointegrated=False,
            adf_statistic=np.nan, p_value=1.0,
            beta=0.0, alpha=0.0, lookback_days=n
        )

    alpha_coeff = float(coeffs[0])
    beta_coeff = float(coeffs[1])
    residuals = y - (alpha_coeff + beta_coeff * x)

    # ADF test on residuals
    try:
        adf_stat, p_value, _, _, _, _ = _adf_test(residuals)
    except Exception as e:
        logger.warning(f"ADF test failed for {leg1}/{leg2}: {e}")
        return CointegrationResult(
            leg1=leg1, leg2=leg2, cointegrated=False,
            adf_statistic=np.nan, p_value=1.0,
            beta=beta_coeff, alpha=alpha_coeff, lookback_days=n
        )

    cointegrated = p_value < p_threshold

    if not cointegrated:
        logger.info(
            f"Cointegration test FAILED: {leg1}/{leg2} | "
            f"ADF={adf_stat:.3f} p={p_value:.4f} (threshold={p_threshold})"
        )

    return CointegrationResult(
        leg1=leg1, leg2=leg2, cointegrated=cointegrated,
        adf_statistic=float(adf_stat), p_value=float(p_value),
        beta=beta_coeff, alpha=alpha_coeff, lookback_days=n
    )


def _adf_test(series: np.ndarray, max_lags: int = 5) -> tuple:
    """
    Augmented Dickey-Fuller test.
    Returns (adf_statistic, p_value, used_lag, nobs, critical_values, icbest).
    Uses statsmodels if available; falls back to simplified version.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        return adfuller(series, maxlag=max_lags, autolag='AIC')
    except ImportError:
        # Simplified ADF (no lag selection) — for environments without statsmodels
        n = len(series)
        diff = np.diff(series)
        lagged = series[:-1]

        # OLS: Δy_t = ρ*y_{t-1} + ε_t
        x = lagged.reshape(-1, 1)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(x, diff, rcond=None)
            rho = float(coeffs[0])
            residuals = diff - rho * lagged
            se = np.sqrt(np.sum(residuals**2) / (n - 2)) / (np.std(lagged) * np.sqrt(n - 1))
            t_stat = rho / (se + 1e-10)
            # Approximate p-value from t-distribution (simplified)
            p_value = float(stats.t.sf(abs(t_stat), df=n - 2) * 2)
            return t_stat, p_value, 0, n - 1, {}, None
        except Exception:
            return 0.0, 1.0, 0, n, {}, None


# ─────────────────────────────────────────────────────────────────────────────
# SPREAD Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────
def compute_spread_zscore(
    prices_y: np.ndarray,
    prices_x: np.ndarray,
    beta: float,
    alpha: float,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling Z-score of the cointegration residual (spread).
    spread_t = Y_t - (alpha + beta * X_t)
    z_t = (spread_t - mean(spread, W)) / std(spread, W)
    """
    n = min(len(prices_y), len(prices_x))
    spread = prices_y[-n:] - (alpha + beta * prices_x[-n:])

    z = np.full(n, np.nan)
    for t in range(window, n):
        w = spread[t - window:t]
        mu = np.mean(w)
        sigma = np.std(w, ddof=1)
        if sigma > 1e-10:
            z[t] = (spread[t] - mu) / sigma

    return z


# ─────────────────────────────────────────────────────────────────────────────
# STAT ARB SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StatArbSignal:
    """
    Signal output for research/paper trading only.
    NEVER passed to order manager while allocation = 0.
    """
    leg1: str
    leg2: str
    direction: SignalDirection
    spread_zscore: float
    cointegration: CointegrationResult
    beta: float
    entry_spread: float
    signal_strength: float
    leg_execution_risk_pips: float  # Estimated execution gap cost
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suspended_reason: Optional[str] = None

    # Mandatory warning — always present
    allocation_warning: str = (
        "ZERO ALLOCATION: Statistical arbitrage is experimental. "
        "Leg execution risk at MT5 speeds represents 20-30% of gross edge. "
        "Capital allocation forbidden until leg execution prerequisites resolved."
    )

    @property
    def is_actionable(self) -> bool:
        """Always False — zero allocation engine cannot act."""
        _check_allocation_guard()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STAT ARB ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class StatisticalArbitrageEngine:
    """
    Alpha Engine 5: Statistical Arbitrage — EXPERIMENTAL

    Research and paper trading use only. Zero capital allocation.
    All signals are generated but NEVER forwarded to order manager.

    Generates signals for monitoring purposes and validates that
    the economic foundation (cointegration) holds in live data.
    """

    # Signal thresholds
    ENTRY_ZSCORE_THRESHOLD = 2.0       # Enter when |z| > 2σ
    EXIT_ZSCORE_THRESHOLD = 0.5        # Exit when |z| < 0.5σ (mean reversion)
    STOP_ZSCORE_THRESHOLD = 3.5        # Stop when |z| > 3.5σ (relationship broken)

    # Cointegration re-test frequency
    RETEST_INTERVAL_BARS = 20          # Re-test every 20 bars

    def __init__(self, config: "SystemConfig"):
        _check_allocation_guard()  # Fail fast if someone enables allocation
        self.config = config

        self._cointegration_cache: dict[str, CointegrationResult] = {}
        self._retest_counter: dict[str, int] = {}

    def compute_signals(
        self,
        bar_arrays: dict[str, "BarArray"],
        lookback_days: int = 180,
    ) -> list[StatArbSignal]:
        """
        Compute stat arb signals for all approved pairs.
        Research/paper trading use only — signals are never executed.

        Parameters
        ----------
        bar_arrays : Dict of {symbol: BarArray}
        lookback_days : Lookback for cointegration test

        Returns
        -------
        List of StatArbSignal (for monitoring only)
        """
        signals = []

        for leg1, leg2 in APPROVED_PAIRS:
            if leg1 not in bar_arrays or leg2 not in bar_arrays:
                logger.debug(f"Missing bar array for pair {leg1}/{leg2}")
                continue

            signal = self._compute_pair_signal(
                bar_arrays[leg1], bar_arrays[leg2],
                leg1, leg2, lookback_days
            )
            signals.append(signal)

        return signals

    def _compute_pair_signal(
        self,
        bar_array_1: "BarArray",
        bar_array_2: "BarArray",
        leg1: str,
        leg2: str,
        lookback_days: int,
    ) -> StatArbSignal:
        """Compute signal for a single pair."""
        pair_key = f"{leg1}_{leg2}"
        n = min(bar_array_1.n, bar_array_2.n, lookback_days)

        if n < 60:
            return self._flat_signal(
                leg1, leg2, f"insufficient_bars({n})"
            )

        prices_1 = bar_array_1.close[-n:]
        prices_2 = bar_array_2.close[-n:]

        # ── Cointegration test (with re-test scheduling) ──────────────────────
        self._retest_counter[pair_key] = self._retest_counter.get(pair_key, 0) + 1
        needs_retest = (
            pair_key not in self._cointegration_cache
            or self._retest_counter[pair_key] >= self.RETEST_INTERVAL_BARS
        )

        if needs_retest:
            coint = engle_granger_cointegration(
                prices_1, prices_2, leg1, leg2, p_threshold=0.01
            )
            self._cointegration_cache[pair_key] = coint
            self._retest_counter[pair_key] = 0
            logger.info(
                f"Cointegration re-test {leg1}/{leg2}: "
                f"cointegrated={coint.cointegrated} p={coint.p_value:.4f}"
            )
        else:
            coint = self._cointegration_cache[pair_key]

        if not coint.cointegrated:
            return self._flat_signal(
                leg1, leg2,
                f"not_cointegrated(p={coint.p_value:.4f})",
                cointegration=coint
            )

        # ── Spread Z-score ────────────────────────────────────────────────────
        z_series = compute_spread_zscore(
            prices_1, prices_2, coint.beta, coint.alpha, window=60
        )
        z = z_series[-1]

        if np.isnan(z):
            return self._flat_signal(
                leg1, leg2, "zscore_nan", cointegration=coint
            )

        # ── Leg execution risk estimate ───────────────────────────────────────
        # At 50-200ms per leg, assume worst case 3 pips on EUR/USD-equivalent
        leg_exec_risk_pips = 3.0  # Conservative — could be 1-3 pips

        # ── Entry logic ───────────────────────────────────────────────────────
        if z < -self.ENTRY_ZSCORE_THRESHOLD:
            direction = SignalDirection.LONG_LEG1_SHORT_LEG2
        elif z > self.ENTRY_ZSCORE_THRESHOLD:
            direction = SignalDirection.SHORT_LEG1_LONG_LEG2
        else:
            return self._flat_signal(
                leg1, leg2,
                f"zscore_below_threshold(z={z:.3f})",
                cointegration=coint
            )

        signal_strength = float(np.clip(
            (abs(z) - self.ENTRY_ZSCORE_THRESHOLD) /
            (self.STOP_ZSCORE_THRESHOLD - self.ENTRY_ZSCORE_THRESHOLD),
            0.0, 1.0
        ))

        spread = float(prices_1[-1] - (coint.alpha + coint.beta * prices_2[-1]))

        logger.info(
            f"[PAPER ONLY] StatArb signal: {leg1}/{leg2} | "
            f"{direction.name} | z={z:.3f} | "
            f"leg_exec_risk={leg_exec_risk_pips:.1f}pips"
        )

        return StatArbSignal(
            leg1=leg1,
            leg2=leg2,
            direction=direction,
            spread_zscore=float(z),
            cointegration=coint,
            beta=coint.beta,
            entry_spread=spread,
            signal_strength=signal_strength,
            leg_execution_risk_pips=leg_exec_risk_pips,
        )

    def should_exit(
        self,
        z_current: float,
        direction: SignalDirection,
    ) -> tuple[bool, str]:
        """Exit logic for paper trading position tracking."""
        if direction == SignalDirection.LONG_LEG1_SHORT_LEG2:
            if z_current >= -self.EXIT_ZSCORE_THRESHOLD:
                return True, f"mean_reversion_complete(z={z_current:.3f})"
            if z_current < -self.STOP_ZSCORE_THRESHOLD:
                return True, f"stop_hit(z={z_current:.3f})"
        elif direction == SignalDirection.SHORT_LEG1_LONG_LEG2:
            if z_current <= self.EXIT_ZSCORE_THRESHOLD:
                return True, f"mean_reversion_complete(z={z_current:.3f})"
            if z_current > self.STOP_ZSCORE_THRESHOLD:
                return True, f"stop_hit(z={z_current:.3f})"
        return False, ""

    def implementation_prerequisites(self) -> dict:
        """
        Returns checklist of prerequisites for stat arb capital allocation.
        For governance tracking — not a trading method.
        """
        return {
            "prerequisites": [
                {
                    "item": "Leg execution solution",
                    "description": (
                        "Broker offers simultaneous basket/OCO execution, OR "
                        "leg risk is explicitly modelled as structural cost from first test"
                    ),
                    "met": False,  # Must be manually verified
                },
                {
                    "item": "12 months paper trading",
                    "description": (
                        "12+ months validated paper trading with "
                        "explicit leg execution risk accounting"
                    ),
                    "met": False,  # Must be manually verified
                },
                {
                    "item": "DSR > 0.95",
                    "description": "Deflated Sharpe Ratio exceeds 0.95",
                    "met": False,
                },
                {
                    "item": "PBO < 0.10",
                    "description": "Probability of Backtest Overfitting < 10%",
                    "met": False,
                },
                {
                    "item": "Human adversarial review",
                    "description": (
                        "Named human adversarial reviewer has attempted to falsify "
                        "and approved for Stage 6"
                    ),
                    "met": False,
                },
            ],
            "all_met": False,
            "allocation_pct": STAT_ARB_ALLOCATION_PCT,
        }

    def _flat_signal(
        self,
        leg1: str,
        leg2: str,
        reason: str,
        cointegration: Optional[CointegrationResult] = None,
    ) -> StatArbSignal:
        if cointegration is None:
            cointegration = CointegrationResult(
                leg1=leg1, leg2=leg2, cointegrated=False,
                adf_statistic=np.nan, p_value=1.0,
                beta=0.0, alpha=0.0, lookback_days=0
            )
        return StatArbSignal(
            leg1=leg1,
            leg2=leg2,
            direction=SignalDirection.FLAT,
            spread_zscore=0.0,
            cointegration=cointegration,
            beta=0.0,
            entry_spread=0.0,
            signal_strength=0.0,
            leg_execution_risk_pips=3.0,
            suspended_reason=reason,
        )
