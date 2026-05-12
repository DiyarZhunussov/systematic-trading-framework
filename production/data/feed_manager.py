"""
feed_manager.py — Data Feed Manager
Handles all data ingestion from MT5, applies validation, and provides
clean OHLCV arrays to the signal engines.

Design principles:
- All data passes through data_validator before use
- In-memory cache only — no disk persistence for live data
- Rollover-adjusted data required for indices (enforced with warning)
- Staleness detection prevents signals from computing on outdated bars
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MT5 import is optional at module load (not available in research environment)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from production.data.data_validator import (
    Bar,
    SeriesValidationResult,
    validate_bar,
    validate_series,
    check_rollover_contamination,
)

# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAME MAPPING: config string → MT5 constant
# ─────────────────────────────────────────────────────────────────────────────
if MT5_AVAILABLE:
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }
else:
    TIMEFRAME_MAP = {}


# ─────────────────────────────────────────────────────────────────────────────
# BAR ARRAY — NumPy-backed price arrays for signal computation
# ─────────────────────────────────────────────────────────────────────────────
class BarArray:
    """
    Fixed-length rolling window of validated bar data.
    Provides NumPy arrays for direct use in signal computation.
    """

    def __init__(self, instrument: str, timeframe: str, max_bars: int = 500):
        self.instrument = instrument
        self.timeframe = timeframe
        self.max_bars = max_bars
        self._bars: list[Bar] = []
        self._last_updated: Optional[datetime] = None

    def update(self, bars: list[Bar]) -> None:
        """Replace internal bar list with freshly validated bars."""
        self._bars = bars[-self.max_bars:]
        self._last_updated = datetime.now(timezone.utc)

    def append(self, bar: Bar) -> None:
        """Append a single new bar (for tick-based updates)."""
        if self._bars and bar.time <= self._bars[-1].time:
            return  # Duplicate or out-of-order — ignore
        self._bars.append(bar)
        if len(self._bars) > self.max_bars:
            self._bars.pop(0)
        self._last_updated = datetime.now(timezone.utc)

    @property
    def n(self) -> int:
        return len(self._bars)

    @property
    def close(self) -> np.ndarray:
        return np.array([b.close for b in self._bars])

    @property
    def open(self) -> np.ndarray:
        return np.array([b.open for b in self._bars])

    @property
    def high(self) -> np.ndarray:
        return np.array([b.high for b in self._bars])

    @property
    def low(self) -> np.ndarray:
        return np.array([b.low for b in self._bars])

    @property
    def volume(self) -> np.ndarray:
        return np.array([b.tick_volume for b in self._bars])

    @property
    def timestamps(self) -> list[datetime]:
        return [b.time for b in self._bars]

    @property
    def latest_bar(self) -> Optional[Bar]:
        return self._bars[-1] if self._bars else None

    @property
    def last_updated(self) -> Optional[datetime]:
        return self._last_updated

    def is_stale(self, max_age_seconds: int) -> bool:
        if self._last_updated is None:
            return True
        age = (datetime.now(timezone.utc) - self._last_updated).total_seconds()
        return age > max_age_seconds

    def returns(self, log: bool = True) -> np.ndarray:
        """Compute return series. log=True for log returns."""
        prices = self.close
        if len(prices) < 2:
            return np.array([])
        if log:
            return np.diff(np.log(prices + 1e-10))
        return np.diff(prices) / (prices[:-1] + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# FEED MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class FeedManager:
    """
    Manages data ingestion from MT5 for all active instruments and timeframes.
    Provides validated BarArray objects to signal engines.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        self._cache: dict[str, BarArray] = {}  # key: "EURUSD_D1"
        self._connected = False
        self._rollover_warned: set[str] = set()

    def _cache_key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument}_{timeframe}"

    # ── Connection ────────────────────────────────────────────────────────────
    def is_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        info = mt5.terminal_info()
        return info is not None and info.connected

    # ── Core data fetch ───────────────────────────────────────────────────────
    def fetch_bars(
        self,
        instrument: str,
        timeframe: str,
        n_bars: int = 300,
        validate: bool = True,
    ) -> Optional[BarArray]:
        """
        Fetch N bars from MT5 for the given instrument and timeframe.
        Validates all bars before returning. Returns None on failure.

        Parameters
        ----------
        instrument : MT5 symbol string (e.g. "EURUSD")
        timeframe : Timeframe string (e.g. "D1", "H4")
        n_bars : Number of bars to fetch (fetches n_bars + 50 for buffer)
        validate : If True, validate the full series before returning

        Returns
        -------
        BarArray or None if fetch/validation fails
        """
        if not MT5_AVAILABLE:
            logger.error("MT5 not available — cannot fetch live bars")
            return None

        tf_const = TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        # Fetch with buffer to ensure we have enough after validation filtering
        fetch_count = n_bars + 50
        raw_rates = mt5.copy_rates_from_pos(instrument, tf_const, 0, fetch_count)

        if raw_rates is None or len(raw_rates) == 0:
            logger.error(
                f"MT5 returned no data for {instrument} {timeframe}. "
                f"Error: {mt5.last_error()}"
            )
            return None

        # Convert to Bar objects
        bars = []
        for r in raw_rates:
            bar = Bar(
                time=datetime.fromtimestamp(r["time"], tz=timezone.utc),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                tick_volume=int(r["tick_volume"]),
                spread=int(r.get("spread", 0)),
                real_volume=int(r.get("real_volume", 0)),
            )
            bars.append(bar)

        if not bars:
            return None

        # ── Rollover contamination check for indices ──────────────────────────
        cache_key = self._cache_key(instrument, timeframe)
        instr_config = self.config.instruments.get("instruments", {}).get(instrument, {})
        asset_class = instr_config.get("asset_class", "forex")

        if asset_class == "index" and cache_key not in self._rollover_warned:
            suspicious = check_rollover_contamination(bars, instrument)
            if suspicious:
                logger.warning(
                    f"{instrument}: Rollover contamination detected. "
                    f"All signals for this instrument use potentially "
                    f"contaminated data until rollover-adjusted feed is confirmed."
                )
                self._rollover_warned.add(cache_key)

        # ── Series validation ─────────────────────────────────────────────────
        if validate:
            series_result = validate_series(
                bars, instrument, timeframe, min_bars=min(50, n_bars // 2)
            )
            if not series_result.valid:
                logger.error(
                    f"Series validation failed for {instrument} {timeframe}: "
                    f"{series_result.error or series_result.failed_indices}"
                )
                return None

            if series_result.warnings:
                for w in series_result.warnings:
                    logger.warning(f"{instrument} {timeframe}: {w}")

        # ── Cache and return ──────────────────────────────────────────────────
        bar_array = BarArray(instrument, timeframe, max_bars=n_bars + 100)
        bar_array.update(bars[-n_bars:])

        self._cache[cache_key] = bar_array
        logger.debug(
            f"Fetched {len(bars)} bars for {instrument} {timeframe} — "
            f"latest bar: {bars[-1].time}"
        )
        return bar_array

    def get_cached(
        self,
        instrument: str,
        timeframe: str,
        max_age_seconds: int = 600,
    ) -> Optional[BarArray]:
        """
        Return cached BarArray if it exists and is not stale.
        Returns None if cache miss or stale — caller should call fetch_bars().
        """
        key = self._cache_key(instrument, timeframe)
        bar_array = self._cache.get(key)
        if bar_array is None:
            return None
        if bar_array.is_stale(max_age_seconds):
            logger.debug(f"Cache stale for {instrument} {timeframe}")
            return None
        return bar_array

    def get_or_fetch(
        self,
        instrument: str,
        timeframe: str,
        n_bars: int = 300,
        max_cache_age_seconds: int = 300,
    ) -> Optional[BarArray]:
        """
        Return cached bars if fresh, otherwise fetch from MT5.
        Primary interface for signal engines.
        """
        cached = self.get_cached(instrument, timeframe, max_cache_age_seconds)
        if cached is not None:
            return cached
        return self.fetch_bars(instrument, timeframe, n_bars)

    def invalidate_cache(self, instrument: Optional[str] = None) -> None:
        """Invalidate cache for a specific instrument or all instruments."""
        if instrument is None:
            self._cache.clear()
            logger.info("Cleared all data cache")
        else:
            keys_to_clear = [k for k in self._cache if k.startswith(instrument)]
            for k in keys_to_clear:
                del self._cache[k]
            logger.info(f"Cleared cache for {instrument}")

    # ── Bulk prefetch ─────────────────────────────────────────────────────────
    def prefetch_all(
        self,
        instruments: list[str],
        timeframes: list[str],
        n_bars: int = 300,
    ) -> dict[str, bool]:
        """
        Prefetch bars for all instrument/timeframe combinations at startup.
        Returns {symbol_tf: success} mapping.
        """
        results = {}
        for instrument in instruments:
            for timeframe in timeframes:
                key = self._cache_key(instrument, timeframe)
                bar_array = self.fetch_bars(instrument, timeframe, n_bars)
                results[key] = bar_array is not None
                if bar_array is None:
                    logger.error(f"Prefetch failed: {instrument} {timeframe}")

        n_success = sum(results.values())
        logger.info(
            f"Prefetch complete: {n_success}/{len(results)} "
            f"instrument/timeframe pairs loaded"
        )
        return results

    # ── Live tick ─────────────────────────────────────────────────────────────
    def get_latest_tick(self, instrument: str) -> Optional[dict]:
        """Return the latest bid/ask tick for an instrument."""
        if not MT5_AVAILABLE:
            return None
        tick = mt5.symbol_info_tick(instrument)
        if tick is None:
            logger.warning(
                f"No tick data for {instrument}. Error: {mt5.last_error()}"
            )
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "volume": tick.volume,
            "flags": tick.flags,
        }

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def cache_summary(self) -> dict:
        """Return a summary of all cached series for monitoring."""
        now = datetime.now(timezone.utc)
        summary = {}
        for key, bar_array in self._cache.items():
            age = None
            if bar_array.last_updated:
                age = (now - bar_array.last_updated).total_seconds()
            summary[key] = {
                "n_bars": bar_array.n,
                "latest_bar": bar_array.latest_bar.time if bar_array.latest_bar else None,
                "cache_age_seconds": age,
            }
        return summary
