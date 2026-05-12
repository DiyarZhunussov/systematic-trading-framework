"""
regime_engine.py — Alpha Engine 4: Regime Adaptation
Implements Section 3, Alpha Engine 4 and Part IV of the framework.

Conceptual status:
    Regime adaptation is NOT an independent alpha source.
    It is a capital allocation and risk adjustment mechanism that modulates
    the other engines based on detected market state.

Primary regime input: ADX (replaces Hurst exponent per adversarial review)
    ADX > 25 → Trending regime → favour trend following
    ADX < 20 → Choppy regime  → favour mean reversion
    20–25    → Transition     → reduce conviction, reduce size on all signals

Crisis detection (multi-factor):
    C1: Equity index realised vol / 252-day median > 2.5
    C2: FX implied vol proxy (hourly ATR z-score) > 2.0
    C3: XAU/USD absolute 24-hour return > 3σ
    C4: VIX-equivalent spike (if observable)
    Crisis ACTIVE: ≥ 2 of C1–C4 simultaneously true
    Crisis RESOLVED: 0 of C1–C4 active for ≥ 3 consecutive days

Hurst exponent:
    Retired as primary signal. Retained as low-weight background context only.
    At 100-period R/S window: estimation variance ≈ ±0.15–0.20 — wider than
    classification thresholds (±0.05 from H=0.5). Not a trading signal.

CRITICAL interaction: ADX signals during active crisis are SUPPRESSED.
    Do not initiate new trend-following positions when crisis indicator is active.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME ENUMS
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityLevel(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRISIS = "crisis"


class TrendState(Enum):
    TRENDING = "trending"
    CHOPPY = "choppy"
    TRANSITION = "transition"


# ─────────────────────────────────────────────────────────────────────────────
# REGIME STATE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RegimeState:
    """
    Full regime classification result.
    Passed to all alpha engines and the risk engine.
    """
    vol_level: VolatilityLevel
    trend_state: TrendState
    crisis_active: bool

    # Raw indicator values
    adx: float
    adx_period: int
    realized_vol_annualised: float
    vol_percentile: float           # Percentile within 252-day window
    hurst_background: float         # Low-weight context only — not a signal

    # Crisis factor flags
    crisis_c1_equity_vol: bool
    crisis_c2_fx_vol: bool
    crisis_c3_gold_return: bool
    crisis_c4_vix_spike: bool
    crisis_factors_active: int

    # Confirmation tracking
    confirmation_bars_remaining: int    # Bars until regime change is confirmed
    pending_vol_level: Optional[VolatilityLevel]
    pending_trend_state: Optional[TrendState]

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def regime_key(self) -> tuple[str, str]:
        """Returns (vol_level, trend_state) tuple for allocation matrix lookup."""
        if self.crisis_active:
            return ("crisis", "any")
        return (self.vol_level.value, self.trend_state.value)

    @property
    def is_transition(self) -> bool:
        return self.trend_state == TrendState.TRANSITION

    def log_summary(self) -> str:
        crisis_str = " [CRISIS]" if self.crisis_active else ""
        return (
            f"Regime: {self.vol_level.value}/{self.trend_state.value}{crisis_str} | "
            f"ADX={self.adx:.1f} | "
            f"vol_pct={self.vol_percentile:.0f} | "
            f"crisis_factors={self.crisis_factors_active}/4"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY SCALES (from allocation matrix — Section 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyScales:
    """
    Scale factors for each alpha engine in the current regime.
    These are proportional multipliers (0–1), not portfolio weights.
    Always passed through normalize_strategy_scales() before use.
    """
    trend_following: float      # Engine 2
    mean_reversion: float       # Engine 1
    volatility_breakout: float  # Engine 3

    def as_dict(self) -> dict:
        return {
            "trend": self.trend_following,
            "mean_rev": self.mean_reversion,
            "breakout": self.volatility_breakout,
        }

    def total(self) -> float:
        return self.trend_following + self.mean_reversion + self.volatility_breakout


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION MATRIX (Section 4)
# ─────────────────────────────────────────────────────────────────────────────
ALLOCATION_MATRIX: dict[tuple[str, str], dict[str, float]] = {
    ("low", "trending"):    {"trend": 1.00, "mean_rev": 0.00, "breakout": 0.00},
    ("low", "choppy"):      {"trend": 0.30, "mean_rev": 0.70, "breakout": 0.20},
    ("low", "transition"):  {"trend": 0.50, "mean_rev": 0.35, "breakout": 0.10},
    ("normal", "trending"): {"trend": 0.80, "mean_rev": 0.10, "breakout": 0.40},
    ("normal", "choppy"):   {"trend": 0.40, "mean_rev": 0.60, "breakout": 0.20},
    ("normal", "transition"):{"trend": 0.60, "mean_rev": 0.35, "breakout": 0.30},
    ("high", "trending"):   {"trend": 0.60, "mean_rev": 0.00, "breakout": 0.30},
    ("high", "choppy"):     {"trend": 0.20, "mean_rev": 0.00, "breakout": 0.50},
    ("high", "transition"): {"trend": 0.40, "mean_rev": 0.00, "breakout": 0.40},
    ("crisis", "any"):      {"trend": 0.30, "mean_rev": 0.00, "breakout": 0.00},
}

# Floor: no strategy below 10% in non-crisis regimes (diversification insurance)
# Exception: mean_rev in high-vol and crisis (categorical failure in crises)
STRATEGY_FLOORS = {
    "low":    {"trend": 0.10, "mean_rev": 0.10, "breakout": 0.10},
    "normal": {"trend": 0.10, "mean_rev": 0.10, "breakout": 0.10},
    "high":   {"trend": 0.10, "mean_rev": 0.00, "breakout": 0.10},  # No floor for MR
    "crisis": {"trend": 0.00, "mean_rev": 0.00, "breakout": 0.00},  # No floors in crisis
}

MAX_AGGREGATE_SCALE = 1.0


def normalize_strategy_scales(scales: dict[str, float]) -> dict[str, float]:
    """
    Normalise strategy scale factors so aggregate never exceeds 1.0.
    Preserves relative proportions.

    Examples:
        {'trend': 0.80, 'mean_rev': 0.10, 'breakout': 0.40} → total 1.30
        → {'trend': 0.615, 'mean_rev': 0.077, 'breakout': 0.308} → total 1.00

        {'trend': 0.60, 'mean_rev': 0.00, 'breakout': 0.30} → total 0.90
        → unchanged (total ≤ 1.0)
    """
    total = sum(scales.values())
    if total <= MAX_AGGREGATE_SCALE:
        return dict(scales)
    factor = MAX_AGGREGATE_SCALE / total
    return {k: round(v * factor, 6) for k, v in scales.items()}


def get_regime_scales(
    vol_level: VolatilityLevel,
    trend_state: TrendState,
    crisis_active: bool,
) -> StrategyScales:
    """
    Return normalised strategy scales for the given regime.
    Applies floors and normalisation.
    """
    if crisis_active:
        key = ("crisis", "any")
        floor_key = "crisis"
    else:
        key = (vol_level.value, trend_state.value)
        floor_key = vol_level.value

    raw = ALLOCATION_MATRIX.get(key, ALLOCATION_MATRIX[("normal", "choppy")])
    floors = STRATEGY_FLOORS.get(floor_key, {"trend": 0.10, "mean_rev": 0.10, "breakout": 0.10})

    # Apply floors — only to strategies with non-zero raw allocation
    floored = {
        k: max(v, floors.get(k, 0.0)) if v > 0 else v
        for k, v in raw.items()
    }

    normalised = normalize_strategy_scales(floored)

    return StrategyScales(
        trend_following=normalised.get("trend", 0.0),
        mean_reversion=normalised.get("mean_rev", 0.0),
        volatility_breakout=normalised.get("breakout", 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ADX, +DI, -DI.
    ADX > 25: trending; ADX < 20: choppy; 20-25: transition.
    Returns (adx, plus_di, minus_di).
    """
    n = len(high)
    if n < period + 1:
        nan = np.full(n, np.nan)
        return nan, nan, nan

    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        tr[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t - 1]),
            abs(low[t] - close[t - 1]),
        )

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for t in range(1, n):
        up = high[t] - high[t - 1]
        down = low[t - 1] - low[t]
        plus_dm[t] = up if (up > down and up > 0) else 0.0
        minus_dm[t] = down if (down > up and down > 0) else 0.0

    # Smoothed with Wilder EMA (period-bar initial sum, then Wilder smoothing)
    def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        result = np.full(n, np.nan)
        result[p] = np.sum(arr[1:p + 1])
        for t in range(p + 1, n):
            result[t] = result[t - 1] - result[t - 1] / p + arr[t]
        return result

    atr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for t in range(period, n):
        if atr_smooth[t] > 1e-10:
            plus_di[t] = 100 * plus_dm_smooth[t] / atr_smooth[t]
            minus_di[t] = 100 * minus_dm_smooth[t] / atr_smooth[t]
            di_sum = plus_di[t] + minus_di[t]
            if di_sum > 1e-10:
                dx[t] = 100 * abs(plus_di[t] - minus_di[t]) / di_sum

    # ADX = Wilder smoothing of DX
    adx = np.full(n, np.nan)
    start = period * 2
    if n > start:
        valid_dx = dx[period:start]
        valid_mask = ~np.isnan(valid_dx)
        if valid_mask.sum() >= period:
            adx[start] = np.mean(valid_dx[valid_mask])
            alpha_w = 1.0 / period
            for t in range(start + 1, n):
                if not np.isnan(dx[t]) and not np.isnan(adx[t - 1]):
                    adx[t] = adx[t - 1] + alpha_w * (dx[t] - adx[t - 1])

    return adx, plus_di, minus_di


def realized_volatility(
    returns: np.ndarray,
    window: int = 20,
    annualise: bool = True,
) -> np.ndarray:
    """Rolling realised volatility (std dev of returns)."""
    n = len(returns)
    rv = np.full(n, np.nan)
    for t in range(window, n):
        rv[t] = np.std(returns[t - window:t], ddof=1)
    if annualise:
        rv = rv * np.sqrt(252)
    return rv


def hurst_exponent_rs(prices: np.ndarray, window: int = 250) -> float:
    """
    Hurst exponent via R/S analysis over trailing window.

    BACKGROUND CONTEXT ONLY — not a trading signal (Section 4).
    At window=100: estimation variance ≈ ±0.15–0.20 (too noisy to use).
    At window=250: lower variance but is a historical label, not real-time.

    Returns NaN if insufficient data or computation fails.
    """
    if len(prices) < window + 1:
        return np.nan

    series = prices[-window:]
    returns = np.diff(np.log(series + 1e-10))
    n = len(returns)
    if n < 20:
        return np.nan

    lags = [max(2, int(n / k)) for k in range(2, min(20, n // 2))]
    lags = sorted(set(lags))

    rs_values = []
    lag_values = []

    for lag in lags:
        if lag >= n:
            continue
        chunks = [returns[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append(np.log(np.mean(rs_chunk)))
            lag_values.append(np.log(lag))

    if len(rs_values) < 4:
        return np.nan

    try:
        hurst, _ = np.polyfit(lag_values, rs_values, 1)
        return float(np.clip(hurst, 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM STRUCTURAL BREAK
# ─────────────────────────────────────────────────────────────────────────────
def cusum_structural_break(
    daily_pnl: np.ndarray,
    false_positive_rate: float = 0.05,
) -> dict:
    """
    CUSUM test for structural break in P&L series.
    S_t = S_{t-1} + (r_t − μ_null) / σ_r

    Alert threshold calibrated at false_positive_rate against null distribution.

    Returns dict: {alert: bool, cusum_value: float, threshold: float}
    """
    if len(daily_pnl) < 20:
        return {"alert": False, "cusum_value": 0.0, "threshold": np.nan,
                "reason": "insufficient_data"}

    # Null: zero mean (strategy has no edge in null hypothesis)
    mu_null = 0.0
    sigma_r = np.std(daily_pnl, ddof=1)

    if sigma_r < 1e-10:
        return {"alert": False, "cusum_value": 0.0, "threshold": np.nan,
                "reason": "zero_variance"}

    # Compute CUSUM
    cusum = np.cumsum((daily_pnl - mu_null) / sigma_r)
    cusum_stat = float(abs(cusum[-1]))

    # Threshold: from asymptotic theory for two-sided CUSUM
    # At 5% level: threshold ≈ 1.36 * sqrt(T)
    T = len(daily_pnl)
    if false_positive_rate <= 0.01:
        k = 1.63
    elif false_positive_rate <= 0.05:
        k = 1.36
    else:
        k = 1.22
    threshold = k * np.sqrt(T)

    alert = cusum_stat > threshold

    if alert:
        logger.warning(
            f"CUSUM structural break alert: "
            f"stat={cusum_stat:.2f} > threshold={threshold:.2f} (T={T})"
        )

    return {
        "alert": alert,
        "cusum_value": cusum_stat,
        "threshold": float(threshold),
        "cusum_series": cusum.tolist(),
        "n_observations": T,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REGIME ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class RegimeEngine:
    """
    Engine 4: Regime Adaptation

    Detects current market regime using ADX (primary), realised volatility
    percentile (secondary), and multi-factor crisis indicators.

    Outputs RegimeState + StrategyScales used by all other engines.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        p = config.regime_params

        # ADX thresholds
        self.adx_trending = float(p.get("adx_trending_threshold", 25.0))
        self.adx_choppy = float(p.get("adx_choppy_threshold", 20.0))
        self.adx_period = int(p.get("adx_period", 14))

        # Volatility percentile thresholds
        self.vol_pct_low = int(p.get("vol_percentile_low", 25))
        self.vol_pct_high = int(p.get("vol_percentile_high", 75))
        self.vol_pct_crisis = int(p.get("vol_percentile_crisis", 90))
        self.vol_lookback = int(p.get("vol_lookback_days", 252))

        # Confirmation days before regime change accepted
        self.confirmation_days = int(p.get("regime_change_confirmation_days", 3))

        # Crisis thresholds
        self.crisis_equity_vol_ratio = float(p.get("crisis_equity_vol_ratio", 2.5))
        self.crisis_fx_vol_zscore = float(p.get("crisis_fx_vol_zscore", 2.0))
        self.crisis_gold_sigma = float(p.get("crisis_gold_return_sigma", 3.0))
        self.crisis_min_indicators = int(p.get("crisis_min_indicators", 2))
        self.crisis_resolution_days = int(p.get("crisis_resolution_days", 3))

        # Internal state
        self._current_regime: Optional[RegimeState] = None
        self._crisis_resolution_counter: int = 0
        self._pending_regime_counter: int = 0
        self._pending_vol: Optional[VolatilityLevel] = None
        self._pending_trend: Optional[TrendState] = None

    # ── Main compute method ───────────────────────────────────────────────────
    def compute_regime(
        self,
        equity_index_bar_array: "BarArray",           # For vol regime + crisis C1
        fx_bar_array: Optional["BarArray"] = None,    # For crisis C2
        gold_bar_array: Optional["BarArray"] = None,  # For crisis C3
        hurst_window: int = 250,
    ) -> RegimeState:
        """
        Compute current regime from market data.

        Parameters
        ----------
        equity_index_bar_array : Primary data source — equity index daily bars
                                  (e.g. SPX500 or NQ100)
        fx_bar_array : Optional — used for FX vol crisis indicator (C2)
        gold_bar_array : Optional — used for XAU crisis indicator (C3)
        hurst_window : Window for background Hurst computation (250 days)

        Returns
        -------
        RegimeState with full regime classification and strategy scales
        """
        prices = equity_index_bar_array.close
        high = equity_index_bar_array.high
        low = equity_index_bar_array.low
        n = len(prices)

        if n < self.vol_lookback + self.adx_period * 2 + 10:
            logger.warning(
                f"Insufficient bars for regime detection ({n} bars). "
                f"Using conservative default: normal/choppy."
            )
            return self._default_regime()

        # ── ADX: trend state ──────────────────────────────────────────────────
        adx_series, _, _ = compute_adx(high, low, prices, self.adx_period)
        adx = float(adx_series[-1]) if not np.isnan(adx_series[-1]) else 22.0

        if adx > self.adx_trending:
            raw_trend_state = TrendState.TRENDING
        elif adx < self.adx_choppy:
            raw_trend_state = TrendState.CHOPPY
        else:
            raw_trend_state = TrendState.TRANSITION

        # ── Realised volatility: vol level ───────────────────────────────────
        returns = np.diff(np.log(prices + 1e-10))
        rv_series = realized_volatility(returns, window=20, annualise=True)
        rv_current = rv_series[-1] if not np.isnan(rv_series[-1]) else 0.15

        rv_lookback = rv_series[-self.vol_lookback:]
        rv_valid = rv_lookback[~np.isnan(rv_lookback)]
        vol_percentile = float(
            np.mean(rv_valid <= rv_current) * 100
            if len(rv_valid) > 10 else 50.0
        )

        if vol_percentile <= self.vol_pct_low:
            raw_vol_level = VolatilityLevel.LOW
        elif vol_percentile >= self.vol_pct_high:
            raw_vol_level = VolatilityLevel.HIGH
        else:
            raw_vol_level = VolatilityLevel.NORMAL

        # ── Crisis detection (multi-factor) ───────────────────────────────────
        c1 = self._crisis_c1_equity_vol(rv_current, rv_valid)
        c2 = self._crisis_c2_fx_vol(fx_bar_array)
        c3 = self._crisis_c3_gold_return(gold_bar_array)
        c4 = False  # VIX proxy — not directly observable via MT5 (placeholder)

        crisis_factors = sum([c1, c2, c3, c4])
        crisis_active = crisis_factors >= self.crisis_min_indicators

        # Crisis resolution requires 3 consecutive days with 0 indicators
        if crisis_active:
            self._crisis_resolution_counter = 0
        else:
            if self._current_regime and self._current_regime.crisis_active:
                self._crisis_resolution_counter += 1
                if self._crisis_resolution_counter < self.crisis_resolution_days:
                    crisis_active = True  # Still in resolution window

        if crisis_active:
            raw_vol_level = VolatilityLevel.CRISIS

        # ── 3-day confirmation smoothing ──────────────────────────────────────
        vol_level, trend_state, pending_vol, pending_trend, confirm_remaining = (
            self._apply_confirmation_smoothing(raw_vol_level, raw_trend_state, crisis_active)
        )

        # ── Hurst (background context — NOT a signal) ─────────────────────────
        hurst = hurst_exponent_rs(prices, window=hurst_window)

        regime = RegimeState(
            vol_level=vol_level,
            trend_state=trend_state,
            crisis_active=crisis_active,
            adx=adx,
            adx_period=self.adx_period,
            realized_vol_annualised=float(rv_current),
            vol_percentile=vol_percentile,
            hurst_background=float(hurst) if not np.isnan(hurst) else 0.5,
            crisis_c1_equity_vol=c1,
            crisis_c2_fx_vol=c2,
            crisis_c3_gold_return=c3,
            crisis_c4_vix_spike=c4,
            crisis_factors_active=crisis_factors,
            confirmation_bars_remaining=confirm_remaining,
            pending_vol_level=pending_vol,
            pending_trend_state=pending_trend,
            timestamp=datetime.now(timezone.utc),
        )

        self._current_regime = regime

        logger.info(f"Regime: {regime.log_summary()}")
        return regime

    def get_strategy_scales(self, regime: RegimeState) -> StrategyScales:
        """
        Return normalised strategy scales for the given regime.
        CRITICAL: ADX signals during crisis are suppressed per Section 4.
        """
        scales = get_regime_scales(
            regime.vol_level,
            regime.trend_state,
            regime.crisis_active,
        )

        # Suppress trend following during transition — reduce conviction
        if regime.is_transition and not regime.crisis_active:
            scales = StrategyScales(
                trend_following=scales.trend_following * 0.7,
                mean_reversion=scales.mean_reversion * 0.7,
                volatility_breakout=scales.volatility_breakout * 0.7,
            )
            logger.debug(
                "Transition regime — all strategy scales reduced 30%"
            )

        return scales

    # ── Crisis indicators ─────────────────────────────────────────────────────
    def _crisis_c1_equity_vol(
        self,
        rv_current: float,
        rv_history: np.ndarray,
    ) -> bool:
        """C1: Equity index realised vol / 252-day median > 2.5"""
        if len(rv_history) < 20:
            return False
        median_rv = float(np.nanmedian(rv_history))
        if median_rv < 1e-10:
            return False
        ratio = rv_current / median_rv
        active = ratio > self.crisis_equity_vol_ratio
        if active:
            logger.warning(f"Crisis C1 active: equity vol ratio = {ratio:.2f}")
        return active

    def _crisis_c2_fx_vol(self, fx_bar_array: Optional["BarArray"]) -> bool:
        """C2: FX implied vol proxy (hourly ATR z-score) > 2.0"""
        if fx_bar_array is None or fx_bar_array.n < 60:
            return False

        high = fx_bar_array.high
        low = fx_bar_array.low
        close = fx_bar_array.close

        # ATR as vol proxy
        from production.engines.signal_engine_mean_reversion import compute_atr
        atr = compute_atr(high, low, close, period=1)  # 1-bar ATR = range

        valid = atr[~np.isnan(atr)]
        if len(valid) < 20:
            return False

        current_atr = valid[-1]
        mean_atr = np.mean(valid[:-1])
        std_atr = np.std(valid[:-1], ddof=1)

        if std_atr < 1e-10:
            return False

        zscore = (current_atr - mean_atr) / std_atr
        active = zscore > self.crisis_fx_vol_zscore
        if active:
            logger.warning(f"Crisis C2 active: FX vol z-score = {zscore:.2f}")
        return active

    def _crisis_c3_gold_return(self, gold_bar_array: Optional["BarArray"]) -> bool:
        """C3: XAU/USD absolute 24-hour return > 3σ"""
        if gold_bar_array is None or gold_bar_array.n < 30:
            return False

        returns = gold_bar_array.returns(log=True)
        if len(returns) < 20:
            return False

        current_ret = abs(returns[-1])
        sigma = np.std(returns[:-1], ddof=1)

        if sigma < 1e-10:
            return False

        zscore = current_ret / sigma
        active = zscore > self.crisis_gold_sigma
        if active:
            logger.warning(
                f"Crisis C3 active: Gold |return| = {current_ret:.4f} "
                f"({zscore:.1f}σ)"
            )
        return active

    # ── Confirmation smoothing ────────────────────────────────────────────────
    def _apply_confirmation_smoothing(
        self,
        raw_vol: VolatilityLevel,
        raw_trend: TrendState,
        crisis_active: bool,
    ) -> tuple[
        VolatilityLevel, TrendState,
        Optional[VolatilityLevel], Optional[TrendState], int
    ]:
        """
        Apply 3-day confirmation before acting on regime changes.
        Returns (confirmed_vol, confirmed_trend, pending_vol, pending_trend, bars_remaining)
        """
        # Crisis overrides confirmation — acts immediately
        if crisis_active:
            self._pending_vol = None
            self._pending_trend = None
            self._pending_regime_counter = 0
            return VolatilityLevel.CRISIS, TrendState.CHOPPY, None, None, 0

        if self._current_regime is None:
            # First computation — accept immediately
            return raw_vol, raw_trend, None, None, 0

        current_vol = self._current_regime.vol_level
        current_trend = self._current_regime.trend_state

        regime_changed = (raw_vol != current_vol) or (raw_trend != current_trend)

        if not regime_changed:
            # Same regime — reset pending
            self._pending_vol = None
            self._pending_trend = None
            self._pending_regime_counter = 0
            return current_vol, current_trend, None, None, 0

        # Track pending regime change
        if self._pending_vol == raw_vol and self._pending_trend == raw_trend:
            self._pending_regime_counter += 1
        else:
            self._pending_vol = raw_vol
            self._pending_trend = raw_trend
            self._pending_regime_counter = 1

        bars_remaining = max(0, self.confirmation_days - self._pending_regime_counter)

        if self._pending_regime_counter >= self.confirmation_days:
            # Confirmed — switch regime
            logger.info(
                f"Regime change confirmed after {self.confirmation_days} days: "
                f"{current_vol.value}/{current_trend.value} → "
                f"{raw_vol.value}/{raw_trend.value}"
            )
            self._pending_vol = None
            self._pending_trend = None
            self._pending_regime_counter = 0
            return raw_vol, raw_trend, None, None, 0
        else:
            # Not yet confirmed — maintain current
            return current_vol, current_trend, raw_vol, raw_trend, bars_remaining

    def _default_regime(self) -> RegimeState:
        """Conservative default when insufficient data."""
        return RegimeState(
            vol_level=VolatilityLevel.NORMAL,
            trend_state=TrendState.CHOPPY,
            crisis_active=False,
            adx=22.0,
            adx_period=self.adx_period,
            realized_vol_annualised=0.15,
            vol_percentile=50.0,
            hurst_background=0.5,
            crisis_c1_equity_vol=False,
            crisis_c2_fx_vol=False,
            crisis_c3_gold_return=False,
            crisis_c4_vix_spike=False,
            crisis_factors_active=0,
            confirmation_bars_remaining=0,
            pending_vol_level=None,
            pending_trend_state=None,
        )

    @property
    def current_regime(self) -> Optional[RegimeState]:
        return self._current_regime
