"""
signal_engine_mean_reversion.py — Alpha Engine 1: Intraday Mean Reversion
Implements Section 3, Alpha Engine 1 of the framework.

Economic foundation:
    Exploits temporary price dislocations from short-term liquidity imbalances.
    Edge is compensation for bearing short-term inventory risk.
    Strategy targets dislocations with reversion timescales of minutes to hours —
    timescales that HFT participants ignore as insufficient for their capacity.

Key design decisions (from adversarial review):
    - Autocorrelation filter is REQUIRED, not optional — suspends signal in
      trending regimes where fading the move has directional risk
    - Intraday volatility normalisation adjusts Z-score for session effects
    - Time stop is mandatory — positions never held beyond T_max hours
    - Capital cap: 15% of risk budget until 12 months validated live trading

Falsification requirements (must pass before any capital allocation):
    - Permutation test: actual Sharpe in top 5% of 1,000 permuted sequences
    - Cost stress: +50% all costs → strategy still profitable
    - Parameter perturbation: ±20% window/threshold → Sharpe degradation < 50%
    - Cross-instrument: works across ≥ 3 instruments
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
class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class MeanReversionSignal:
    """Output of the mean reversion engine for a single instrument."""
    instrument: str
    direction: SignalDirection
    z_score: float                  # Raw Z-score at signal time
    z_score_adj: float              # Session-volatility-adjusted Z-score
    autocorr_rho1: float            # First-order autocorrelation (filter value)
    autocorr_active: bool           # True = mean-reverting regime confirmed
    entry_price: float              # Expected entry price (latest close)
    stop_distance_pips: float       # ATR-based stop distance
    signal_strength: float          # |z_adj| normalised to [0, 1] for IC tracking
    session_vol_ratio: float        # Current session vol / global session vol
    timestamp: datetime
    suspended_reason: Optional[str] = None  # If direction=FLAT, why

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.FLAT

    def log_summary(self) -> str:
        return (
            f"{self.instrument} | {self.direction.name} | "
            f"z={self.z_score:.3f} z_adj={self.z_score_adj:.3f} | "
            f"rho1={self.autocorr_rho1:.3f} active={self.autocorr_active} | "
            f"strength={self.signal_strength:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# All use only past data at each point — no lookahead
# ─────────────────────────────────────────────────────────────────────────────
def rolling_zscore(
    prices: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute rolling Z-score: z_t = (Close_t − μ_t) / σ_t
    where μ and σ are computed over trailing window.

    Uses only past data at each point — no lookahead.
    Returns array of same length as prices (NaN for first window-1 bars).
    """
    n = len(prices)
    z = np.full(n, np.nan)
    for t in range(window, n):
        window_prices = prices[t - window:t]
        mu = np.mean(window_prices)
        sigma = np.std(window_prices, ddof=1)
        if sigma > 1e-10:
            z[t] = (prices[t] - mu) / sigma
    return z


def rolling_autocorrelation(
    prices: np.ndarray,
    window: int,
    lag: int = 1,
) -> np.ndarray:
    """
    Compute rolling lag-1 autocorrelation of returns.
    ρ_1 = Corr(r_t, r_{t-1}) over trailing window.

    Signal ACTIVE only if ρ_1 < -0.10 (mean-reverting confirmed).
    Signal SUSPENDED if ρ_1 > +0.10 (trending — do not fade the move).
    """
    returns = np.diff(prices)
    n = len(returns)
    rho = np.full(len(prices), np.nan)
    for t in range(window + lag, n + 1):
        r_window = returns[t - window:t]
        if len(r_window) < window:
            continue
        r_t = r_window[lag:]
        r_lag = r_window[:-lag]
        if np.std(r_t) > 1e-10 and np.std(r_lag) > 1e-10:
            rho[t] = float(np.corrcoef(r_t, r_lag)[0, 1])
    return rho


def session_volatility_profile(
    prices: np.ndarray,
    timestamps: list[datetime],
    lookback_bars: int = 60 * 24 * 60,  # 60 days at 1-min bars
) -> dict[int, float]:
    """
    Compute median absolute return per UTC hour over lookback.
    σ_session(h) = median |r_t| for bars in hour h.

    Returns dict: {hour_utc: median_abs_return}
    Used for intraday volatility normalisation of Z-scores.
    """
    if len(prices) != len(timestamps):
        return {}

    hourly_returns: dict[int, list[float]] = {h: [] for h in range(24)}
    returns = np.abs(np.diff(prices[-lookback_bars:]))
    ts = timestamps[-lookback_bars:]

    for i, ret in enumerate(returns):
        hour = ts[i].hour
        hourly_returns[hour].append(ret)

    profile = {}
    for hour, rets in hourly_returns.items():
        if rets:
            profile[hour] = float(np.median(rets))
        else:
            profile[hour] = np.nan

    return profile


def adjust_zscore_for_session(
    z_raw: float,
    current_hour: int,
    session_profile: dict[int, float],
) -> float:
    """
    Adjust Z-score for session volatility differences.
    z_adj = z_raw × (σ_global / σ_session_current)

    Prevents over-signalling in low-volatility sessions (Asian)
    and under-signalling in high-volatility sessions (London/NY overlap).
    """
    if not session_profile:
        return z_raw

    valid_vols = [v for v in session_profile.values() if not np.isnan(v) and v > 0]
    if not valid_vols:
        return z_raw

    sigma_global = float(np.median(valid_vols))
    sigma_session = session_profile.get(current_hour, np.nan)

    if np.isnan(sigma_session) or sigma_session <= 1e-10:
        return z_raw

    # Clip adjustment to prevent extreme values in unusual sessions
    ratio = np.clip(sigma_global / sigma_session, 0.25, 4.0)
    return float(z_raw * ratio)


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20,
) -> np.ndarray:
    """
    Compute Average True Range.
    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = EMA(TR, period)
    """
    n = len(high)
    tr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for t in range(1, n):
        tr[t] = max(
            high[t] - low[t],
            abs(high[t] - close[t - 1]),
            abs(low[t] - close[t - 1]),
        )

    # EMA of TR
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        alpha = 2.0 / (period + 1)
        for t in range(period, n):
            atr[t] = alpha * tr[t] + (1 - alpha) * atr[t - 1]

    return atr


# ─────────────────────────────────────────────────────────────────────────────
# MEAN REVERSION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class MeanReversionEngine:
    """
    Alpha Engine 1: Intraday Mean Reversion

    Usage:
        engine = MeanReversionEngine(config)
        signal = engine.compute_signal(bar_array, instrument)
        if signal.is_actionable:
            # pass to risk engine for sizing
    """

    # Autocorrelation filter thresholds (Section 3, Engine 1)
    AUTOCORR_MEAN_REVERTING_THRESHOLD = -0.10   # ρ₁ < this → active
    AUTOCORR_TRENDING_THRESHOLD = +0.10          # ρ₁ > this → suspend

    def __init__(self, config: "SystemConfig"):
        self.config = config
        # Per-instrument session volatility profiles (updated periodically)
        self._session_profiles: dict[str, dict[int, float]] = {}

    def _get_instrument_params(self, instrument: str) -> dict:
        """Return per-instrument parameters with safe defaults."""
        try:
            return self.config.instrument(instrument)
        except KeyError:
            logger.warning(f"No config for {instrument} — using defaults")
            return {
                "mean_reversion_threshold": 2.0,
                "mean_reversion_window": 30,
                "atr_period": 20,
                "time_stop_hours": 3,
            }

    def update_session_profile(
        self,
        instrument: str,
        prices: np.ndarray,
        timestamps: list[datetime],
    ) -> None:
        """
        Update the session volatility profile for an instrument.
        Should be called once per day (or on cache refresh).
        """
        profile = session_volatility_profile(prices, timestamps)
        self._session_profiles[instrument] = profile
        logger.debug(
            f"Session profile updated for {instrument}: "
            f"{sum(1 for v in profile.values() if not np.isnan(v))} hours"
        )

    def compute_signal(
        self,
        bar_array: "BarArray",
        instrument: str,
        regime_scale: float = 1.0,
    ) -> MeanReversionSignal:
        """
        Compute mean reversion signal for the latest bar.

        Parameters
        ----------
        bar_array : Validated BarArray from FeedManager
        instrument : Symbol string
        regime_scale : Scale factor from regime engine (0-1)

        Returns
        -------
        MeanReversionSignal — direction=FLAT if no actionable signal
        """
        params = self._get_instrument_params(instrument)
        window = params.get("mean_reversion_window", 30)
        threshold = params.get("mean_reversion_threshold", 2.0)
        atr_period = params.get("atr_period", 20)

        prices = bar_array.close
        timestamps = bar_array.timestamps
        high = bar_array.high
        low = bar_array.low
        n = len(prices)

        # Need enough bars for all indicators
        min_required = max(window * 2, atr_period + 10)
        if n < min_required:
            return self._flat_signal(
                instrument, prices[-1] if n > 0 else 0.0,
                f"insufficient_bars({n} < {min_required})"
            )

        # ── Compute Z-score ───────────────────────────────────────────────────
        z_series = rolling_zscore(prices, window)
        z_raw = z_series[-1]

        if np.isnan(z_raw):
            return self._flat_signal(
                instrument, prices[-1],
                "z_score_nan"
            )

        # ── Session volatility adjustment ─────────────────────────────────────
        # Update profile if not yet computed for this instrument
        if instrument not in self._session_profiles:
            self.update_session_profile(instrument, prices, timestamps)

        current_hour = datetime.now(timezone.utc).hour
        session_profile = self._session_profiles.get(instrument, {})
        z_adj = adjust_zscore_for_session(z_raw, current_hour, session_profile)

        # Session vol ratio for signal metadata
        valid_vols = [
            v for v in session_profile.values()
            if not np.isnan(v) and v > 0
        ]
        sigma_global = float(np.median(valid_vols)) if valid_vols else 1.0
        sigma_session = session_profile.get(current_hour, sigma_global)
        session_vol_ratio = (
            sigma_session / sigma_global
            if sigma_global > 1e-10 else 1.0
        )

        # ── Autocorrelation filter ────────────────────────────────────────────
        rho_series = rolling_autocorrelation(prices, window)
        rho1 = rho_series[-1]

        if np.isnan(rho1):
            return self._flat_signal(
                instrument, prices[-1],
                "autocorr_nan"
            )

        autocorr_active = rho1 < self.AUTOCORR_MEAN_REVERTING_THRESHOLD
        autocorr_trending = rho1 > self.AUTOCORR_TRENDING_THRESHOLD

        # ── ATR for stop distance ─────────────────────────────────────────────
        atr_series = compute_atr(high, low, prices, atr_period)
        atr = atr_series[-1]

        if np.isnan(atr) or atr <= 0:
            return self._flat_signal(
                instrument, prices[-1],
                "atr_invalid"
            )

        # Stop is at z_adj > threshold * 1.5 (extended dislocation)
        stop_distance = atr * 1.5

        # ── Signal strength for IC tracking ──────────────────────────────────
        # Normalise |z_adj| to [0,1] — at threshold=2.0, strength≈0.5
        signal_strength = float(np.clip(abs(z_adj) / (threshold * 2), 0.0, 1.0))

        # ── Entry logic ───────────────────────────────────────────────────────
        entry_price = float(prices[-1])

        # Suspend if regime_scale is effectively zero
        if regime_scale < 0.05:
            return self._flat_signal(
                instrument, entry_price,
                f"regime_scale_too_low({regime_scale:.3f})",
                z_score=z_raw, z_adj=z_adj,
                rho1=rho1, autocorr_active=autocorr_active,
                stop_distance=stop_distance,
                signal_strength=signal_strength,
                session_vol_ratio=session_vol_ratio,
            )

        # Suspend if trending regime detected
        if autocorr_trending:
            return self._flat_signal(
                instrument, entry_price,
                "trending_regime(rho1>{:.2f})".format(
                    self.AUTOCORR_TRENDING_THRESHOLD
                ),
                z_score=z_raw, z_adj=z_adj,
                rho1=rho1, autocorr_active=False,
                stop_distance=stop_distance,
                signal_strength=signal_strength,
                session_vol_ratio=session_vol_ratio,
            )

        # Require mean-reverting autocorrelation
        if not autocorr_active:
            return self._flat_signal(
                instrument, entry_price,
                f"autocorr_neutral(rho1={rho1:.3f})",
                z_score=z_raw, z_adj=z_adj,
                rho1=rho1, autocorr_active=False,
                stop_distance=stop_distance,
                signal_strength=signal_strength,
                session_vol_ratio=session_vol_ratio,
            )

        # Check Z-score threshold
        if z_adj < -threshold:
            direction = SignalDirection.LONG
        elif z_adj > threshold:
            direction = SignalDirection.SHORT
        else:
            return self._flat_signal(
                instrument, entry_price,
                f"z_below_threshold(z_adj={z_adj:.3f}, threshold={threshold:.1f})",
                z_score=z_raw, z_adj=z_adj,
                rho1=rho1, autocorr_active=autocorr_active,
                stop_distance=stop_distance,
                signal_strength=signal_strength,
                session_vol_ratio=session_vol_ratio,
            )

        signal = MeanReversionSignal(
            instrument=instrument,
            direction=direction,
            z_score=z_raw,
            z_score_adj=z_adj,
            autocorr_rho1=rho1,
            autocorr_active=autocorr_active,
            entry_price=entry_price,
            stop_distance_pips=stop_distance,
            signal_strength=signal_strength,
            session_vol_ratio=session_vol_ratio,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(f"MR signal: {signal.log_summary()}")
        return signal

    # ── Exit logic ────────────────────────────────────────────────────────────
    def should_exit(
        self,
        instrument: str,
        entry_z_adj: float,
        current_prices: np.ndarray,
        current_timestamps: list[datetime],
        entry_timestamp: datetime,
        direction: SignalDirection,
    ) -> tuple[bool, str]:
        """
        Determine whether an open mean reversion position should be closed.

        Exit conditions (any one triggers):
        1. Z_adj crosses zero (primary exit — mean reversion complete)
        2. Z_adj exceeds threshold * 1.5 in same direction (stop — trend taking over)
        3. Time limit exceeded (T_max hours)

        Returns (should_exit: bool, reason: str)
        """
        params = self._get_instrument_params(instrument)
        window = params.get("mean_reversion_window", 30)
        threshold = params.get("mean_reversion_threshold", 2.0)
        time_stop_hours = params.get("time_stop_hours", 3)

        # ── Time stop ─────────────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        if entry_timestamp.tzinfo is None:
            entry_timestamp = entry_timestamp.replace(tzinfo=timezone.utc)
        hours_held = (now - entry_timestamp).total_seconds() / 3600
        if hours_held >= time_stop_hours:
            return True, f"time_stop({hours_held:.1f}h >= {time_stop_hours}h)"

        # ── Z-score based exits ───────────────────────────────────────────────
        z_series = rolling_zscore(current_prices, window)
        z_raw = z_series[-1]
        if np.isnan(z_raw):
            return False, ""

        session_profile = self._session_profiles.get(instrument, {})
        current_hour = now.hour
        z_adj = adjust_zscore_for_session(z_raw, current_hour, session_profile)

        # Primary exit: Z crosses zero
        if direction == SignalDirection.LONG and z_adj >= 0:
            return True, f"z_crossed_zero(z_adj={z_adj:.3f})"
        if direction == SignalDirection.SHORT and z_adj <= 0:
            return True, f"z_crossed_zero(z_adj={z_adj:.3f})"

        # Stop exit: Z extended in same direction (trend taking over)
        stop_threshold = threshold * 1.5
        if direction == SignalDirection.LONG and z_adj < -stop_threshold:
            return True, f"z_extended_stop(z_adj={z_adj:.3f} < -{stop_threshold:.1f})"
        if direction == SignalDirection.SHORT and z_adj > stop_threshold:
            return True, f"z_extended_stop(z_adj={z_adj:.3f} > {stop_threshold:.1f})"

        return False, ""

    # ── Falsification tests ───────────────────────────────────────────────────
    def permutation_test(
        self,
        trade_pnls: list[float],
        n_permutations: int = 1000,
        significance_level: float = 0.05,
    ) -> dict:
        """
        Permutation test on trade sequence.
        Actual Sharpe must be in top 5% of 1,000 permuted sequences.

        Returns dict with: actual_sharpe, percentile, passes
        """
        if len(trade_pnls) < 20:
            return {
                "actual_sharpe": np.nan,
                "percentile": np.nan,
                "passes": False,
                "reason": f"insufficient_trades({len(trade_pnls)} < 20)",
            }

        pnls = np.array(trade_pnls)
        actual_sharpe = (
            np.mean(pnls) / np.std(pnls) * np.sqrt(252)
            if np.std(pnls) > 1e-10 else 0.0
        )

        permuted_sharpes = []
        for _ in range(n_permutations):
            perm = np.random.permutation(pnls)
            s = (
                np.mean(perm) / np.std(perm) * np.sqrt(252)
                if np.std(perm) > 1e-10 else 0.0
            )
            permuted_sharpes.append(s)

        permuted_sharpes = np.array(permuted_sharpes)
        percentile = float(np.mean(permuted_sharpes < actual_sharpe))
        passes = percentile >= (1.0 - significance_level)

        return {
            "actual_sharpe": float(actual_sharpe),
            "percentile": percentile,
            "passes": passes,
            "threshold": 1.0 - significance_level,
        }

    def parameter_sensitivity_test(
        self,
        prices: np.ndarray,
        instrument: str,
        perturbation: float = 0.20,
    ) -> dict:
        """
        Perturb window and threshold by ±perturbation.
        Sharpe degradation must remain < 50% across all perturbations.

        Returns summary of Sharpe ratios across parameter variants.
        """
        params = self._get_instrument_params(instrument)
        base_window = params.get("mean_reversion_window", 30)
        base_threshold = params.get("mean_reversion_threshold", 2.0)

        results = {}
        windows = [
            int(base_window * (1 - perturbation)),
            base_window,
            int(base_window * (1 + perturbation)),
        ]
        thresholds = [
            base_threshold * (1 - perturbation),
            base_threshold,
            base_threshold * (1 + perturbation),
        ]

        sharpes = []
        for w in windows:
            for t in thresholds:
                if w < 5:
                    continue
                z = rolling_zscore(prices, w)
                # Simulate long when z < -t, short when z > t, flat otherwise
                positions = np.zeros(len(prices))
                for i in range(len(z)):
                    if np.isnan(z[i]):
                        continue
                    if z[i] < -t:
                        positions[i] = 1
                    elif z[i] > t:
                        positions[i] = -1

                ret = np.diff(prices) / (prices[:-1] + 1e-10)
                strategy_ret = positions[:-1] * ret
                if np.std(strategy_ret) > 1e-10:
                    sr = np.mean(strategy_ret) / np.std(strategy_ret) * np.sqrt(252)
                else:
                    sr = 0.0
                sharpes.append(sr)
                results[f"w{w}_t{t:.2f}"] = float(sr)

        if not sharpes:
            return {"passes": False, "reason": "no_valid_combinations"}

        base_sharpe = results.get(f"w{base_window}_t{base_threshold:.2f}", 0.0)
        min_sharpe = min(sharpes)
        degradation = (
            (base_sharpe - min_sharpe) / abs(base_sharpe)
            if abs(base_sharpe) > 1e-10 else 1.0
        )

        return {
            "base_sharpe": base_sharpe,
            "min_sharpe": min_sharpe,
            "max_degradation_pct": float(degradation * 100),
            "passes": degradation < 0.50,
            "variants": results,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _flat_signal(
        self,
        instrument: str,
        entry_price: float,
        reason: str,
        z_score: float = 0.0,
        z_adj: float = 0.0,
        rho1: float = 0.0,
        autocorr_active: bool = False,
        stop_distance: float = 0.0,
        signal_strength: float = 0.0,
        session_vol_ratio: float = 1.0,
    ) -> MeanReversionSignal:
        return MeanReversionSignal(
            instrument=instrument,
            direction=SignalDirection.FLAT,
            z_score=z_score,
            z_score_adj=z_adj,
            autocorr_rho1=rho1,
            autocorr_active=autocorr_active,
            entry_price=entry_price,
            stop_distance_pips=stop_distance,
            signal_strength=signal_strength,
            session_vol_ratio=session_vol_ratio,
            timestamp=datetime.now(timezone.utc),
            suspended_reason=reason,
        )
