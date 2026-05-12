"""
data_validator.py — Data Validation Pipeline
Validates OHLCV bars before any signal computation.

Key principles (Section 9.5):
- Validate timestamp recency, OHLC consistency, range sanity, and volume
- Rollover contamination checks for equity index CFDs
- Stale data detection — never compute signals on stale bars
- All failures logged with instrument + bar details for forensic audit
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BAR DATACLASS
# Canonical in-memory representation of a single OHLCV bar
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Bar:
    """Single OHLCV bar."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: Optional[int] = None
    real_volume: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "time": self.time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tick_volume": self.tick_volume,
            "spread": self.spread,
            "real_volume": self.real_volume,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    """Result of a single bar validation."""
    valid: bool
    instrument: str
    timeframe: str
    bar_time: Optional[datetime]
    failed_checks: list[str]
    warnings: list[str]

    def __bool__(self):
        return self.valid


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED BAR INTERVALS (seconds)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}

# Maximum expected bar ranges per instrument (Section 9.5)
# These are tuned for daily bars; scaled for shorter TFs in validation logic
MAX_BAR_RANGES = {
    "EURUSD": 0.0200,
    "GBPUSD": 0.0250,
    "USDJPY": 2.00,
    "XAUUSD": 50.0,
    "NQ100": 500.0,
    "SPX500": 100.0,
    "DAX40": 500.0,
}

# Scale factors for intraday ranges relative to daily
TIMEFRAME_RANGE_SCALE = {
    "M1": 0.05,
    "M5": 0.10,
    "M15": 0.15,
    "M30": 0.20,
    "H1": 0.30,
    "H4": 0.55,
    "D1": 1.00,
    "W1": 2.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BAR VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────
def validate_bar(
    bar: Bar,
    instrument: str,
    timeframe: str,
    staleness_multiplier: float = 3.0,
) -> ValidationResult:
    """
    Validate a single OHLCV bar before signal computation.

    Parameters
    ----------
    bar : Bar dataclass
    instrument : Symbol string (e.g. "EURUSD")
    timeframe : MT5 timeframe string (e.g. "M5", "D1")
    staleness_multiplier : Bar is stale if age > interval * multiplier

    Returns
    -------
    ValidationResult — valid=True only if ALL checks pass
    """
    failed = []
    warnings = []
    now_utc = datetime.now(timezone.utc)

    # Ensure bar time is timezone-aware
    bar_time = bar.time
    if bar_time.tzinfo is None:
        bar_time = bar_time.replace(tzinfo=timezone.utc)

    # ── Check 1: Timestamp recency ────────────────────────────────────────────
    interval_seconds = TIMEFRAME_SECONDS.get(timeframe, 86400)
    bar_age_seconds = (now_utc - bar_time).total_seconds()
    max_age = interval_seconds * staleness_multiplier
    if bar_age_seconds > max_age:
        failed.append(
            f"stale_data(age={bar_age_seconds:.0f}s, max={max_age:.0f}s)"
        )

    # ── Check 2: OHLC logical consistency ────────────────────────────────────
    ohlc_ok = (
        bar.low > 0
        and bar.low <= bar.open <= bar.high
        and bar.low <= bar.close <= bar.high
        and bar.high >= bar.low
    )
    if not ohlc_ok:
        failed.append(
            f"ohlc_inconsistent("
            f"O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close})"
        )

    # ── Check 3: No zero or negative prices ───────────────────────────────────
    for price_name, price_val in [
        ("open", bar.open),
        ("high", bar.high),
        ("low", bar.low),
        ("close", bar.close),
    ]:
        if price_val <= 0:
            failed.append(f"non_positive_price({price_name}={price_val})")

    # ── Check 4: Range sanity — flash crash / data feed spike detection ───────
    max_daily_range = MAX_BAR_RANGES.get(instrument)
    if max_daily_range is not None:
        tf_scale = TIMEFRAME_RANGE_SCALE.get(timeframe, 1.0)
        max_range = max_daily_range * tf_scale
        bar_range = bar.high - bar.low
        if bar_range > max_range:
            failed.append(
                f"range_exceeded("
                f"range={bar_range:.5f}, max={max_range:.5f})"
            )
    else:
        warnings.append(f"unknown_instrument_range({instrument})")

    # ── Check 5: Tick volume ──────────────────────────────────────────────────
    if bar.tick_volume is not None:
        if bar.tick_volume <= 0:
            failed.append(f"zero_tick_volume({bar.tick_volume})")
        elif bar.tick_volume < 5:
            warnings.append(f"low_tick_volume({bar.tick_volume})")

    # ── Check 6: Open equals previous close (gap detection, warning only) ─────
    # This is informational — gaps are valid but worth flagging
    # Implemented at the series level below

    valid = len(failed) == 0

    if not valid:
        logger.warning(
            f"Bar validation FAILED | {instrument} {timeframe} @ {bar_time} | "
            f"checks={failed}"
        )
    elif warnings:
        logger.debug(
            f"Bar validation warnings | {instrument} {timeframe} @ {bar_time} | "
            f"warnings={warnings}"
        )

    return ValidationResult(
        valid=valid,
        instrument=instrument,
        timeframe=timeframe,
        bar_time=bar_time,
        failed_checks=failed,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SERIES VALIDATOR
# Validates a sequence of bars for completeness and consistency
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SeriesValidationResult:
    valid: bool
    instrument: str
    timeframe: str
    n_bars: int
    n_failed: int
    n_gaps: int
    n_spikes: int
    failed_indices: list[int]
    warnings: list[str]
    error: Optional[str] = None


def validate_series(
    bars: list[Bar],
    instrument: str,
    timeframe: str,
    min_bars: int = 50,
    max_gap_multiple: float = 2.5,
    spike_zscore_threshold: float = 5.0,
) -> SeriesValidationResult:
    """
    Validate a series of bars for use in indicator computation.

    Checks:
    - Minimum bar count
    - Individual bar validity
    - Timestamp gaps (missing bars)
    - Price spike detection (Z-score based)
    - Rollover contamination warning for indices (large overnight gaps)

    Parameters
    ----------
    bars : Ordered list of Bar objects (oldest first)
    instrument : Symbol string
    timeframe : MT5 timeframe string
    min_bars : Minimum bars required
    max_gap_multiple : Flag gap if interval > expected * multiplier
    spike_zscore_threshold : Z-score above which a close is flagged as spike

    Returns
    -------
    SeriesValidationResult
    """
    if not bars:
        return SeriesValidationResult(
            valid=False, instrument=instrument, timeframe=timeframe,
            n_bars=0, n_failed=0, n_gaps=0, n_spikes=0,
            failed_indices=[], warnings=[],
            error="empty_series",
        )

    n_bars = len(bars)
    warnings = []

    # ── Minimum bars ──────────────────────────────────────────────────────────
    if n_bars < min_bars:
        return SeriesValidationResult(
            valid=False, instrument=instrument, timeframe=timeframe,
            n_bars=n_bars, n_failed=0, n_gaps=0, n_spikes=0,
            failed_indices=[], warnings=[],
            error=f"insufficient_bars({n_bars} < {min_bars})",
        )

    # ── Individual bar validation ─────────────────────────────────────────────
    failed_indices = []
    for i, bar in enumerate(bars):
        result = validate_bar(bar, instrument, timeframe)
        if not result.valid:
            failed_indices.append(i)

    # ── Gap detection ─────────────────────────────────────────────────────────
    interval_seconds = TIMEFRAME_SECONDS.get(timeframe, 86400)
    n_gaps = 0
    for i in range(1, n_bars):
        t_prev = bars[i - 1].time
        t_curr = bars[i].time
        # Make timezone-aware if needed
        if t_prev.tzinfo is None:
            t_prev = t_prev.replace(tzinfo=timezone.utc)
        if t_curr.tzinfo is None:
            t_curr = t_curr.replace(tzinfo=timezone.utc)
        gap = (t_curr - t_prev).total_seconds()
        if gap > interval_seconds * max_gap_multiple:
            n_gaps += 1
            # Rollover warning for indices (expected around 22:00 UTC weekdays)
            asset_class = _infer_asset_class(instrument)
            if asset_class == "index":
                warnings.append(
                    f"possible_rollover_gap at bar {i} "
                    f"({bars[i].time}) — ensure rollover-adjusted data"
                )
            else:
                logger.debug(
                    f"Gap detected in {instrument} {timeframe}: "
                    f"{gap:.0f}s at bar {i}"
                )

    # ── Spike detection (Z-score on close returns) ────────────────────────────
    closes = np.array([b.close for b in bars])
    returns = np.diff(np.log(closes + 1e-10))
    n_spikes = 0
    if len(returns) > 10:
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        if ret_std > 0:
            zscores = np.abs((returns - ret_mean) / ret_std)
            spike_mask = zscores > spike_zscore_threshold
            n_spikes = int(np.sum(spike_mask))
            if n_spikes > 0:
                warnings.append(
                    f"price_spikes_detected({n_spikes} bars above "
                    f"{spike_zscore_threshold}σ)"
                )

    # ── Decision ──────────────────────────────────────────────────────────────
    # Fail if >5% of bars are individually invalid
    fail_rate = len(failed_indices) / n_bars
    valid = (
        fail_rate <= 0.05
        and n_gaps <= max(2, n_bars * 0.02)  # Allow up to 2% gaps
    )

    if not valid:
        logger.warning(
            f"Series validation FAILED | {instrument} {timeframe} | "
            f"n_bars={n_bars}, failed={len(failed_indices)}, gaps={n_gaps}"
        )

    return SeriesValidationResult(
        valid=valid,
        instrument=instrument,
        timeframe=timeframe,
        n_bars=n_bars,
        n_failed=len(failed_indices),
        n_gaps=n_gaps,
        n_spikes=n_spikes,
        failed_indices=failed_indices,
        warnings=warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROLLOVER ADJUSTMENT CHECK
# Warns if an equity index series appears to contain unadjusted rollover gaps
# ─────────────────────────────────────────────────────────────────────────────
def check_rollover_contamination(
    bars: list[Bar],
    instrument: str,
    overnight_gap_threshold: float = 0.005,  # 0.5% overnight gap flag
) -> list[int]:
    """
    Identify potential rollover contamination in equity index CFD data.

    Equity index CFDs carry financing costs that create artificial overnight
    price gaps in unadjusted data (Section 2.4). These contaminate z-scores,
    ATR calculations, and all normalised signals.

    Returns indices of bars with suspicious overnight gaps.
    Caller should ensure rollover-adjusted data is used before signal computation.
    """
    asset_class = _infer_asset_class(instrument)
    if asset_class != "index":
        return []

    suspicious = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        curr_open = bars[i].open
        if prev_close <= 0:
            continue
        gap_pct = abs(curr_open - prev_close) / prev_close
        if gap_pct > overnight_gap_threshold:
            suspicious.append(i)
            logger.debug(
                f"Possible rollover contamination: {instrument} bar {i} "
                f"open={curr_open:.2f} vs prev_close={prev_close:.2f} "
                f"gap={gap_pct*100:.2f}%"
            )

    if suspicious:
        logger.warning(
            f"{instrument}: {len(suspicious)} potential rollover gaps detected. "
            f"Ensure rollover-adjusted OHLCV data is used for all signal computation."
        )

    return suspicious


# ─────────────────────────────────────────────────────────────────────────────
# TICK VALIDATION
# For the MT5 bridge — validate live tick before order execution
# ─────────────────────────────────────────────────────────────────────────────
def validate_tick(
    bid: float,
    ask: float,
    instrument: str,
    instrument_config: dict,
) -> tuple[bool, str]:
    """
    Validate a live MT5 tick before order submission.

    Returns (is_valid, reason_if_not)
    """
    if bid <= 0 or ask <= 0:
        return False, f"non_positive_price(bid={bid}, ask={ask})"

    if ask < bid:
        return False, f"inverted_spread(bid={bid}, ask={ask})"

    # Spread check
    max_spread = instrument_config.get("max_spread_points")
    if max_spread is not None:
        point = instrument_config.get("point_value", 0.0001)
        spread_points = (ask - bid) / point
        if spread_points > max_spread:
            return (
                False,
                f"spread_too_wide({spread_points:.1f} pts > {max_spread} pts)"
            )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def _infer_asset_class(instrument: str) -> str:
    """Infer asset class from instrument symbol for validation logic."""
    upper = instrument.upper()
    if any(idx in upper for idx in ["NQ", "SPX", "DAX", "DOW", "FTSE", "CAC", "NIKKEI"]):
        return "index"
    if "XAU" in upper or "XAG" in upper or "OIL" in upper:
        return "commodity"
    return "forex"
