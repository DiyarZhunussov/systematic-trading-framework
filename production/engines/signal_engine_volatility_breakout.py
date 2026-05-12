"""
signal_engine_volatility_breakout.py — Alpha Engine 3: Volatility Breakout
Implements Section 3, Alpha Engine 3 of the framework.

Economic foundation:
    Exploits the compression-expansion cycle. Volatility is mean-reverting at
    medium timescales. Periods of extreme low volatility reliably precede elevated
    volatility. Institutional flow constrained during thin markets enters when
    vol expands, amplifying the initial move.

CRITICAL epistemic warning (Section 3):
    This engine is the MOST susceptible to backtest overfitting of the five engines.
    Visually compelling breakout patterns + unlimited parameter combinations =
    systematic data-mining illusions.

    DSR requirement: > 0.95 (vs 0.85 for other engines)
    Capital cap: 10% of risk budget until 12 months validated live trading
    PBO must be < 0.10 before any allocation

Two-signal architecture:
    Signal 1: ATR volatility ratio (VR = ATR(5)/ATR(20)) — compression detector
    Signal 2: Bollinger Band width squeeze — secondary confirmation
    Entry: Both signals active + N_confirm bars elapsed + price breakout
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BreakoutSignal:
    """Output of the volatility breakout engine for a single instrument."""
    instrument: str
    direction: SignalDirection
    vr_ratio: float                 # ATR(5)/ATR(20) — contraction if < 0.70
    bb_width: float                 # Bollinger Band width — squeeze if < P20
    bb_width_percentile: float      # Percentile of current BB width (60-day)
    compression_active: bool        # Both signals confirm compression
    bars_in_compression: int        # Consecutive bars since compression started
    breakout_level: float           # Price level that triggers entry
    direction_is_long: bool
    atr: float
    profit_target: float            # Entry ± (ATR × PT_multiplier)
    stop_loss: float                # Entry ∓ (ATR × SL_multiplier)
    signal_strength: float
    timestamp: datetime
    suspended_reason: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.FLAT

    def log_summary(self) -> str:
        return (
            f"{self.instrument} | {self.direction.name} | "
            f"VR={self.vr_ratio:.3f} BB_pct={self.bb_width_percentile:.1f} | "
            f"compress_bars={self.bars_in_compression} | "
            f"breakout@{self.breakout_level:.5f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """Average True Range — consistent with other engines."""
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


def atr_volatility_ratio(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fast_period: int = 5,
    slow_period: int = 20,
) -> np.ndarray:
    """
    VR_t = ATR(fast) / ATR(slow)
    Contraction signal: VR_t < 0.70
    """
    atr_fast = compute_atr(high, low, close, fast_period)
    atr_slow = compute_atr(high, low, close, slow_period)
    with np.errstate(divide='ignore', invalid='ignore'):
        vr = np.where(atr_slow > 1e-10, atr_fast / atr_slow, np.nan)
    return vr


def bollinger_band_width(
    prices: np.ndarray,
    period: int = 20,
    n_std: float = 2.0,
) -> np.ndarray:
    """
    BB_width = (BB_upper - BB_lower) / BB_middle
    where BB_middle = SMA(period), bands = middle ± n_std × σ

    Returns width as a fraction of the middle band.
    """
    n = len(prices)
    width = np.full(n, np.nan)
    for t in range(period, n):
        window = prices[t - period:t]
        middle = np.mean(window)
        std = np.std(window, ddof=1)
        if middle > 1e-10:
            width[t] = (2 * n_std * std) / middle
    return width


def bb_width_percentile(
    bb_width: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """
    Rolling percentile of BB width over lookback period.
    Squeeze: percentile < 20 (BB_width in bottom 20% of recent history).
    """
    n = len(bb_width)
    pct = np.full(n, np.nan)
    for t in range(lookback, n):
        window = bb_width[t - lookback:t]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        current = bb_width[t]
        if np.isnan(current):
            continue
        pct[t] = float(np.mean(valid <= current) * 100)
    return pct


def rolling_max_high(high: np.ndarray, period: int) -> np.ndarray:
    """Rolling maximum of high prices over past N bars (no lookahead)."""
    n = len(high)
    result = np.full(n, np.nan)
    for t in range(period, n):
        result[t] = np.max(high[t - period:t])
    return result


def rolling_min_low(low: np.ndarray, period: int) -> np.ndarray:
    """Rolling minimum of low prices over past N bars (no lookahead)."""
    n = len(low)
    result = np.full(n, np.nan)
    for t in range(period, n):
        result[t] = np.min(low[t - period:t])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BREAKOUT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityBreakoutEngine:
    """
    Alpha Engine 3: Volatility Breakout

    WARN: Most overfitting-prone engine. DSR threshold 0.95, capital cap 10%.
    Do not allocate capital until full validation pipeline is complete.

    Two-phase logic:
        Phase 1 (compression): VR < 0.70 AND BB width < P20, sustained N_confirm bars
        Phase 2 (breakout): Price breaks above/below N_break bar high/low with VR rising
    """

    # Compression thresholds
    VR_CONTRACTION_THRESHOLD = 0.70
    BB_SQUEEZE_PERCENTILE = 20.0

    def __init__(self, config: "SystemConfig"):
        self.config = config
        # Track compression state per instrument
        self._compression_state: dict[str, dict] = {}

    def _get_instrument_params(self, instrument: str) -> dict:
        try:
            return self.config.instrument(instrument)
        except KeyError:
            logger.warning(f"No config for {instrument} — using defaults")
            return {
                "atr_period": 20,
                "breakout_confirm_bars": 5,
                "breakout_lookback_bars": 10,
            }

    def compute_signal(
        self,
        bar_array: "BarArray",
        instrument: str,
        regime_scale: float = 1.0,
    ) -> BreakoutSignal:
        """
        Compute volatility breakout signal for the latest bar.

        Parameters
        ----------
        bar_array : Validated BarArray (4H bars recommended)
        instrument : Symbol string
        regime_scale : Scale from regime engine

        Returns
        -------
        BreakoutSignal
        """
        params = self._get_instrument_params(instrument)
        atr_period = params.get("atr_period", 20)
        n_confirm = params.get("breakout_confirm_bars", 5)
        n_break = params.get("breakout_lookback_bars", 10)

        # Profit/stop multipliers (framework spec)
        pt_multiplier = 2.0    # ATR multiplier for profit target
        sl_multiplier = 1.0    # ATR multiplier for stop loss

        prices = bar_array.close
        high = bar_array.high
        low = bar_array.low
        n = len(prices)

        min_required = max(atr_period * 3, n_break + n_confirm + 60 + 10)
        if n < min_required:
            return self._flat_signal(
                instrument, prices[-1] if n > 0 else 0.0,
                f"insufficient_bars({n} < {min_required})"
            )

        if regime_scale < 0.05:
            return self._flat_signal(
                instrument, prices[-1],
                f"regime_scale_too_low({regime_scale:.3f})"
            )

        # ── Signal 1: ATR volatility ratio ────────────────────────────────────
        vr_series = atr_volatility_ratio(high, low, prices, fast_period=5, slow_period=20)
        vr = vr_series[-1]

        if np.isnan(vr):
            return self._flat_signal(instrument, prices[-1], "vr_nan")

        vr_contraction = vr < self.VR_CONTRACTION_THRESHOLD
        vr_rising = (
            not np.isnan(vr_series[-2])
            and vr > vr_series[-2]
        )

        # ── Signal 2: Bollinger Band width squeeze ────────────────────────────
        bb_width_series = bollinger_band_width(prices, period=20, n_std=2.0)
        bb_pct_series = bb_width_percentile(bb_width_series, lookback=60)

        bb_width_now = bb_width_series[-1]
        bb_pct_now = bb_pct_series[-1]

        if np.isnan(bb_width_now) or np.isnan(bb_pct_now):
            return self._flat_signal(instrument, prices[-1], "bb_nan")

        bb_squeeze = bb_pct_now < self.BB_SQUEEZE_PERCENTILE

        # ── Compression state tracking ────────────────────────────────────────
        compression_active = vr_contraction and bb_squeeze
        state = self._compression_state.get(instrument, {
            "bars_in_compression": 0,
            "compression_started": False,
        })

        if compression_active:
            state["bars_in_compression"] += 1
            state["compression_started"] = True
        else:
            if state.get("compression_started") and not compression_active:
                # Compression ended without breakout — reset after 2 bars
                state["bars_in_compression"] = max(
                    0, state["bars_in_compression"] - 1
                )
            if state["bars_in_compression"] == 0:
                state["compression_started"] = False

        self._compression_state[instrument] = state
        bars_in_compression = state["bars_in_compression"]

        # ── Gate: minimum compression duration ───────────────────────────────
        if bars_in_compression < n_confirm:
            return self._flat_signal(
                instrument, prices[-1],
                f"compression_insufficient({bars_in_compression} < {n_confirm} bars)",
                vr=vr, bb_width=bb_width_now, bb_pct=bb_pct_now,
                compression_active=compression_active,
                bars_in_compression=bars_in_compression,
            )

        # ── Breakout entry logic ──────────────────────────────────────────────
        atr_series = compute_atr(high, low, prices, atr_period)
        atr = atr_series[-1]

        if np.isnan(atr) or atr <= 0:
            return self._flat_signal(instrument, prices[-1], "atr_invalid")

        # Breakout levels: max/min of last N_break bars
        breakout_high = rolling_max_high(high, n_break)[-1]
        breakout_low = rolling_min_low(low, n_break)[-1]

        if np.isnan(breakout_high) or np.isnan(breakout_low):
            return self._flat_signal(
                instrument, prices[-1],
                "breakout_level_nan"
            )

        current_close = float(prices[-1])

        # Long breakout: Close > max(High, N_break bars) AND VR rising
        # Short breakout: Close < min(Low, N_break bars) AND VR rising
        long_breakout = (current_close > breakout_high) and vr_rising
        short_breakout = (current_close < breakout_low) and vr_rising

        if not long_breakout and not short_breakout:
            return self._flat_signal(
                instrument, current_close,
                "no_breakout_yet",
                vr=vr, bb_width=bb_width_now, bb_pct=bb_pct_now,
                compression_active=compression_active,
                bars_in_compression=bars_in_compression,
            )

        direction = SignalDirection.LONG if long_breakout else SignalDirection.SHORT

        # Profit target and stop loss
        if direction == SignalDirection.LONG:
            profit_target = current_close + atr * pt_multiplier
            stop_loss = current_close - atr * sl_multiplier
            breakout_level = breakout_high
        else:
            profit_target = current_close - atr * pt_multiplier
            stop_loss = current_close + atr * sl_multiplier
            breakout_level = breakout_low

        # Signal strength — VR drop depth and BB squeeze depth
        vr_strength = np.clip(
            (self.VR_CONTRACTION_THRESHOLD - vr) / self.VR_CONTRACTION_THRESHOLD,
            0.0, 1.0
        )
        bb_strength = np.clip(
            (self.BB_SQUEEZE_PERCENTILE - bb_pct_now) / self.BB_SQUEEZE_PERCENTILE,
            0.0, 1.0
        ) if bb_pct_now < self.BB_SQUEEZE_PERCENTILE else 0.0
        signal_strength = float((vr_strength + bb_strength) / 2)

        # Reset compression state after breakout signal
        self._compression_state[instrument] = {
            "bars_in_compression": 0,
            "compression_started": False,
        }

        signal = BreakoutSignal(
            instrument=instrument,
            direction=direction,
            vr_ratio=float(vr),
            bb_width=float(bb_width_now),
            bb_width_percentile=float(bb_pct_now),
            compression_active=compression_active,
            bars_in_compression=bars_in_compression,
            breakout_level=float(breakout_level),
            direction_is_long=long_breakout,
            atr=float(atr),
            profit_target=float(profit_target),
            stop_loss=float(stop_loss),
            signal_strength=signal_strength,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"Breakout signal: {signal.log_summary()}")
        return signal

    # ── Exit logic ────────────────────────────────────────────────────────────
    def should_exit(
        self,
        instrument: str,
        direction: SignalDirection,
        entry_price: float,
        current_price: float,
        atr_at_entry: float,
        bars_held: int,
        n_time_stop_bars: int = 20,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Check exit conditions for an open breakout position.
        1. Profit target hit
        2. Stop loss hit
        3. Time stop (N_time bars without hitting either target or stop)
        """
        if direction == SignalDirection.LONG:
            profit_target = entry_price + atr_at_entry * pt_multiplier
            stop_level = entry_price - atr_at_entry * sl_multiplier
            if current_price >= profit_target:
                return True, f"profit_target_hit({current_price:.5f} >= {profit_target:.5f})"
            if current_price <= stop_level:
                return True, f"stop_hit({current_price:.5f} <= {stop_level:.5f})"

        elif direction == SignalDirection.SHORT:
            profit_target = entry_price - atr_at_entry * pt_multiplier
            stop_level = entry_price + atr_at_entry * sl_multiplier
            if current_price <= profit_target:
                return True, f"profit_target_hit({current_price:.5f} <= {profit_target:.5f})"
            if current_price >= stop_level:
                return True, f"stop_hit({current_price:.5f} >= {stop_level:.5f})"

        if bars_held >= n_time_stop_bars:
            return True, f"time_stop({bars_held} bars >= {n_time_stop_bars})"

        return False, ""

    # ── Overfitting warning ───────────────────────────────────────────────────
    def regime_correlation_test(
        self,
        breakout_pnls: list[float],
        trending_regime_flags: list[bool],
    ) -> dict:
        """
        Test whether breakout performance is explained by trend regime
        rather than the breakout signal itself.

        If Sharpe(breakout | trending) ≈ Sharpe(breakout | all) then
        the strategy is trend-regime-driven, not breakout-signal-driven.
        This is an overfitting indicator specific to this engine.
        """
        if len(breakout_pnls) < 30 or len(breakout_pnls) != len(trending_regime_flags):
            return {"passes": False, "reason": "insufficient_data"}

        pnls = np.array(breakout_pnls)
        flags = np.array(trending_regime_flags)

        def sharpe(arr):
            if len(arr) < 5 or np.std(arr) < 1e-10:
                return 0.0
            return float(np.mean(arr) / np.std(arr) * np.sqrt(252))

        sr_all = sharpe(pnls)
        sr_trending = sharpe(pnls[flags])
        sr_non_trending = sharpe(pnls[~flags])

        # If SR in trending regime >> SR overall, performance is regime-driven
        regime_explained = (
            len(pnls[flags]) > 10
            and sr_trending > sr_all * 1.5
            and sr_non_trending < sr_all * 0.5
        )

        return {
            "sharpe_all": sr_all,
            "sharpe_trending": sr_trending,
            "sharpe_non_trending": sr_non_trending,
            "regime_explained": regime_explained,
            "passes": not regime_explained,
            "warning": (
                "Performance appears regime-driven, not breakout-driven"
                if regime_explained else None
            ),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _flat_signal(
        self,
        instrument: str,
        entry_price: float,
        reason: str,
        vr: float = 0.0,
        bb_width: float = 0.0,
        bb_pct: float = 50.0,
        compression_active: bool = False,
        bars_in_compression: int = 0,
    ) -> BreakoutSignal:
        return BreakoutSignal(
            instrument=instrument,
            direction=SignalDirection.FLAT,
            vr_ratio=vr,
            bb_width=bb_width,
            bb_width_percentile=bb_pct,
            compression_active=compression_active,
            bars_in_compression=bars_in_compression,
            breakout_level=0.0,
            direction_is_long=False,
            atr=0.0,
            profit_target=0.0,
            stop_loss=0.0,
            signal_strength=0.0,
            timestamp=datetime.now(timezone.utc),
            suspended_reason=reason,
        )
