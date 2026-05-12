"""
signal_engine_trend_following.py — Alpha Engine 2: Trend Following
Implements Section 3, Alpha Engine 2 of the framework.

Economic foundation:
    Most robustly documented systematic approach with 40+ years of live-money
    evidence. Primary justification is CRISIS ALPHA — not unconditional Sharpe.

    TSMOM decomposition (Kim, Tse & Wald 2016 — key adversarial review finding):
    - Monthly alpha drops from 1.27% WITH vol scaling to 0.41% WITHOUT.
    - Unconditional alpha is therefore partly from vol scaling, not pure momentum.
    - Revised IC estimate: 0.03–0.05 net of vol-scaling decomposition.
    - PRIMARY justification: conditional performance during equity tail events.
      FX and XAU trend following generates positive returns during equity crashes
      because it has no equity-market exposure. Buy-and-hold cannot replicate this.

Signal architecture:
    1. Dual EMA crossover (direction + strength via ATR normalisation)
    2. Donchian channel confirmation (entry timing)
    3. 12-month momentum bias (daily bars only — position skew, not entry timing)

Allocation by trend strength:
    |T_str| > 1.0 → full allocation
    0.5–1.0       → 50% allocation
    < 0.5         → no new entries
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrendSignal:
    """Output of the trend following engine for a single instrument."""
    instrument: str
    direction: "SignalDirection"
    trend_strength: float           # T_str = (EMA_fast - EMA_slow) / ATR(20)
    allocation_scale: float         # 0, 0.5, or 1.0 based on |T_str|
    donchian_confirmed: bool        # True if price confirms Donchian breakout
    momentum_12m: float             # 12-month momentum (MOM_12)
    momentum_bias: float            # Directional bias from MOM_12 in [-1, 1]
    ema_fast: float
    ema_slow: float
    donchian_upper: float
    donchian_lower: float
    entry_price: float
    atr: float
    signal_strength: float          # For IC tracking
    timestamp: datetime
    suspended_reason: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != SignalDirection.FLAT
            and self.allocation_scale > 0
            and self.donchian_confirmed
        )

    def log_summary(self) -> str:
        return (
            f"{self.instrument} | {self.direction.name} | "
            f"T_str={self.trend_strength:.3f} scale={self.allocation_scale:.1f} | "
            f"donchian={self.donchian_confirmed} | mom12={self.momentum_12m:.3f}"
        )


# Re-import to avoid circular dependency
class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def exponential_moving_average(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Compute EMA using standard formula: EMA_t = α*P_t + (1-α)*EMA_{t-1}
    where α = 2/(period+1).
    Returns array same length as prices (NaN for first period-1 bars).
    """
    n = len(prices)
    ema = np.full(n, np.nan)
    if n < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(prices[:period])
    for t in range(period, n):
        ema[t] = alpha * prices[t] + (1 - alpha) * ema[t - 1]
    return ema


def donchian_channel(
    high: np.ndarray,
    low: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Donchian channel: upper = rolling max(High, N), lower = rolling min(Low, N).
    Uses only past bars — does NOT include current bar's high/low.
    Returns (upper, lower).
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for t in range(period, n):
        # Use t-period to t-1 (past bars only — no lookahead)
        upper[t] = np.max(high[t - period:t])
        lower[t] = np.min(low[t - period:t])
    return upper, lower


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """Average True Range — shared utility (same as in mean reversion engine)."""
    n = len(high)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        tr[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t - 1]),
            abs(low[t] - close[t - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        alpha = 2.0 / (period + 1)
        for t in range(period, n):
            atr[t] = alpha * tr[t] + (1 - alpha) * atr[t - 1]
    return atr


def momentum_12m(closes_daily: np.ndarray) -> float:
    """
    12-month time-series momentum: MOM_12 = Close_t / Close_{t-252} − 1.
    Requires at least 253 daily bars.
    Returns NaN if insufficient data.
    """
    if len(closes_daily) < 253:
        return np.nan
    return float(closes_daily[-1] / closes_daily[-253] - 1)


def momentum_bias_weight(mom_12: float, annual_vol: float) -> float:
    """
    Convert 12-month momentum to a position bias in [-1, 1].
    Scaled by annual vol: clip(|MOM_12| / σ_annual, 0, 1) × sign(MOM_12).
    """
    if np.isnan(mom_12) or annual_vol <= 0:
        return 0.0
    raw = np.clip(abs(mom_12) / (annual_vol + 1e-10), 0.0, 1.0)
    return float(np.sign(mom_12) * raw)


# ─────────────────────────────────────────────────────────────────────────────
# TREND FOLLOWING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class TrendFollowingEngine:
    """
    Alpha Engine 2: Trend Following

    Primary signal: Dual EMA crossover + Donchian channel confirmation.
    Secondary signal: 12-month momentum for position bias (not entry timing).

    Note on IC estimate: 0.03–0.05 after Kim et al. (2016) decomposition.
    The true edge is crisis alpha — do not expect consistent unconditional Sharpe.
    """

    # Trend strength allocation thresholds (Section 3, Engine 2)
    FULL_ALLOCATION_THRESHOLD = 1.0
    HALF_ALLOCATION_THRESHOLD = 0.5

    def __init__(self, config: "SystemConfig"):
        self.config = config

    def _get_instrument_params(self, instrument: str) -> dict:
        try:
            return self.config.instrument(instrument)
        except KeyError:
            logger.warning(f"No config for {instrument} — using defaults")
            return {
                "trend_fast_ema": 20,
                "trend_slow_ema": 100,
                "donchian_period_daily": 20,
                "donchian_period_4h": 60,
                "atr_period": 20,
            }

    def compute_signal(
        self,
        bar_array: "BarArray",
        instrument: str,
        timeframe: str = "D1",
        regime_scale: float = 1.0,
        is_daily: bool = True,
    ) -> TrendSignal:
        """
        Compute trend following signal for the latest bar.

        Parameters
        ----------
        bar_array : Validated BarArray (daily bars recommended)
        instrument : Symbol string
        timeframe : Timeframe string — affects Donchian period selection
        regime_scale : Scale from regime engine (0-1)
        is_daily : If True, compute 12-month momentum (requires 253 bars)

        Returns
        -------
        TrendSignal — is_actionable=False if no confirmed trend entry
        """
        params = self._get_instrument_params(instrument)
        n_fast = params.get("trend_fast_ema", 20)
        n_slow = params.get("trend_slow_ema", 100)
        atr_period = params.get("atr_period", 20)

        # Select Donchian period based on timeframe
        if timeframe in ("D1", "W1"):
            donchian_period = params.get("donchian_period_daily", 20)
        else:
            donchian_period = params.get("donchian_period_4h", 60)

        prices = bar_array.close
        high = bar_array.high
        low = bar_array.low
        n = len(prices)

        min_required = n_slow + donchian_period + 10
        if n < min_required:
            return self._flat_signal(
                instrument, prices[-1] if n > 0 else 0.0,
                f"insufficient_bars({n} < {min_required})"
            )

        # ── Dual EMA crossover ────────────────────────────────────────────────
        ema_fast_series = exponential_moving_average(prices, n_fast)
        ema_slow_series = exponential_moving_average(prices, n_slow)

        ema_fast = ema_fast_series[-1]
        ema_slow = ema_slow_series[-1]

        if np.isnan(ema_fast) or np.isnan(ema_slow):
            return self._flat_signal(
                instrument, prices[-1],
                "ema_nan"
            )

        trend_signal_raw = np.sign(ema_fast - ema_slow)  # +1 or -1

        # ── Trend strength (ATR-normalised EMA spread) ────────────────────────
        atr_series = compute_atr(high, low, prices, atr_period)
        atr = atr_series[-1]

        if np.isnan(atr) or atr <= 0:
            return self._flat_signal(
                instrument, prices[-1],
                "atr_invalid"
            )

        trend_strength = float((ema_fast - ema_slow) / atr)

        # ── Allocation scale by trend strength ────────────────────────────────
        abs_strength = abs(trend_strength)
        if abs_strength >= self.FULL_ALLOCATION_THRESHOLD:
            allocation_scale = 1.0
        elif abs_strength >= self.HALF_ALLOCATION_THRESHOLD:
            allocation_scale = 0.5
        else:
            # Below minimum — no new entries
            return self._flat_signal(
                instrument, prices[-1],
                f"trend_strength_insufficient(|T_str|={abs_strength:.3f})",
                trend_strength=trend_strength,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                atr=atr,
            )

        # ── Donchian channel confirmation ─────────────────────────────────────
        d_upper, d_lower = donchian_channel(high, low, donchian_period)
        upper = d_upper[-1]
        lower = d_lower[-1]
        current_close = prices[-1]

        if np.isnan(upper) or np.isnan(lower):
            return self._flat_signal(
                instrument, current_close,
                "donchian_nan"
            )

        # Long: Close > Donchian upper AND EMA crossover bullish
        # Short: Close < Donchian lower AND EMA crossover bearish
        long_confirmed = (current_close > upper) and (trend_signal_raw > 0)
        short_confirmed = (current_close < lower) and (trend_signal_raw < 0)
        donchian_confirmed = long_confirmed or short_confirmed

        # ── 12-month momentum (daily bars only) ───────────────────────────────
        mom_12 = np.nan
        mom_bias = 0.0
        if is_daily and n >= 253:
            mom_12 = momentum_12m(prices)
            annual_returns = np.diff(prices[-253:]) / (prices[-253:-1] + 1e-10)
            annual_vol = float(np.std(annual_returns) * np.sqrt(252))
            mom_bias = momentum_bias_weight(mom_12, annual_vol)

        # ── Direction ─────────────────────────────────────────────────────────
        if not donchian_confirmed:
            direction = SignalDirection.FLAT
            # Not a suspension — just no confirmed entry yet
            # Existing positions can continue; no new entries
        elif long_confirmed:
            direction = SignalDirection.LONG
        else:
            direction = SignalDirection.SHORT

        # ── Regime scale gate ─────────────────────────────────────────────────
        if regime_scale < 0.05 and direction != SignalDirection.FLAT:
            return self._flat_signal(
                instrument, current_close,
                f"regime_scale_too_low({regime_scale:.3f})",
                trend_strength=trend_strength,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                atr=atr,
            )

        signal_strength = float(
            np.clip(abs_strength / (self.FULL_ALLOCATION_THRESHOLD * 2), 0.0, 1.0)
        )

        signal = TrendSignal(
            instrument=instrument,
            direction=direction,
            trend_strength=trend_strength,
            allocation_scale=allocation_scale,
            donchian_confirmed=donchian_confirmed,
            momentum_12m=float(mom_12) if not np.isnan(mom_12) else 0.0,
            momentum_bias=mom_bias,
            ema_fast=float(ema_fast),
            ema_slow=float(ema_slow),
            donchian_upper=float(upper),
            donchian_lower=float(lower),
            entry_price=float(current_close),
            atr=float(atr),
            signal_strength=signal_strength,
            timestamp=datetime.now(timezone.utc),
        )

        if signal.is_actionable:
            logger.info(f"Trend signal: {signal.log_summary()}")

        return signal

    # ── Crisis behaviour ──────────────────────────────────────────────────────
    def crisis_position_check(
        self,
        existing_direction: SignalDirection,
        instrument: str,
        bar_array: "BarArray",
        wider_stop_multiplier: float = 1.5,
    ) -> tuple[bool, float]:
        """
        During crisis regime, existing trend positions may continue with wider stops.
        No new entries. Returns (maintain_position, wider_stop_atr_multiple).

        Per Section 4: Crisis allocation 30%* — existing positions only.
        """
        if existing_direction == SignalDirection.FLAT:
            return False, 0.0

        params = self._get_instrument_params(instrument)
        atr_period = params.get("atr_period", 20)
        prices = bar_array.close
        high = bar_array.high
        low = bar_array.low

        atr_series = compute_atr(high, low, prices, atr_period)
        atr = atr_series[-1]
        if np.isnan(atr) or atr <= 0:
            return False, 0.0

        return True, atr * wider_stop_multiplier

    # ── Falsification tests ───────────────────────────────────────────────────
    def cross_market_consistency(
        self,
        instrument_results: dict[str, dict],
        min_positive_instruments: int = 4,
    ) -> dict:
        """
        Verify trend following works across ≥ 4 of 7 instruments.
        instrument_results: {symbol: {"sharpe": float, "n_trades": int}}
        """
        positive = [
            sym for sym, res in instrument_results.items()
            if res.get("sharpe", 0) > 0 and res.get("n_trades", 0) >= 30
        ]
        passes = len(positive) >= min_positive_instruments

        return {
            "n_positive_instruments": len(positive),
            "positive_instruments": positive,
            "required": min_positive_instruments,
            "passes": passes,
        }

    def performance_surface_check(
        self,
        prices: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        fast_range: tuple[int, int] = (10, 30),
        slow_range: tuple[int, int] = (50, 150),
        n_steps: int = 5,
    ) -> dict:
        """
        Verify Sharpe surface is a broad hill (robust) not a sharp peak (overfit).
        Tests combinations of (N_fast, N_slow) and measures Sharpe variance.

        Passes if: max_sharpe / median_sharpe < 3.0 (broad hill, not spike)
        """
        sharpes = {}
        fast_values = np.linspace(fast_range[0], fast_range[1], n_steps, dtype=int)
        slow_values = np.linspace(slow_range[0], slow_range[1], n_steps, dtype=int)

        for n_fast in fast_values:
            for n_slow in slow_values:
                if n_fast >= n_slow:
                    continue
                ema_f = exponential_moving_average(prices, int(n_fast))
                ema_s = exponential_moving_average(prices, int(n_slow))

                # Simple signal: sign of crossover
                signal = np.sign(ema_f - ema_s)
                ret = np.diff(prices) / (prices[:-1] + 1e-10)
                strat_ret = signal[:-1] * ret
                valid = ~np.isnan(strat_ret)
                if valid.sum() < 30:
                    continue
                sr = strat_ret[valid]
                if np.std(sr) > 1e-10:
                    sharpe = float(np.mean(sr) / np.std(sr) * np.sqrt(252))
                else:
                    sharpe = 0.0
                sharpes[f"f{n_fast}_s{n_slow}"] = sharpe

        if not sharpes:
            return {"passes": False, "reason": "no_valid_combinations"}

        sharpe_values = list(sharpes.values())
        max_s = max(sharpe_values)
        med_s = np.median(sharpe_values)
        ratio = max_s / (abs(med_s) + 1e-10)

        return {
            "max_sharpe": float(max_s),
            "median_sharpe": float(med_s),
            "peak_to_median_ratio": float(ratio),
            "passes": ratio < 3.0,  # Broad hill = robust
            "n_combinations": len(sharpes),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _flat_signal(
        self,
        instrument: str,
        entry_price: float,
        reason: str,
        trend_strength: float = 0.0,
        ema_fast: float = 0.0,
        ema_slow: float = 0.0,
        atr: float = 0.0,
    ) -> TrendSignal:
        return TrendSignal(
            instrument=instrument,
            direction=SignalDirection.FLAT,
            trend_strength=trend_strength,
            allocation_scale=0.0,
            donchian_confirmed=False,
            momentum_12m=0.0,
            momentum_bias=0.0,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            donchian_upper=0.0,
            donchian_lower=0.0,
            entry_price=entry_price,
            atr=atr,
            signal_strength=0.0,
            timestamp=datetime.now(timezone.utc),
            suspended_reason=reason,
        )
