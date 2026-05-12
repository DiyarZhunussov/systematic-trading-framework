"""
decay_monitor.py — Alpha Decay Detection and Retirement
Implements Section 10.5 of the framework.

Daily monitoring metrics (five decay signals):
    1. Rolling 30-day Sharpe vs historical Sharpe
    2. Rolling IC trend (30-day vs prior 30-day)
    3. Slippage deterioration vs 90-day baseline
    4. CUSUM structural break on P&L series
    5. Turnover instability (coefficient of variation)

Alert conditions (Section 6.6):
    A: Rolling 30-day Sharpe < 50% of historical
    B: Rolling 30-day IC < 0 for ≥ 10 consecutive trading days
    C: Regime indicator hostile for > 20 consecutive days
    D: Slippage deterioration > 20% vs baseline
    E: CUSUM structural break alert active

Response schedule by number of active conditions:
    0: Normal operations
    1: Reduce allocation 75%, increase monitoring frequency
    2: Reduce allocation 50%, formal review within 5 days
    3: Suspend strategy — no new entries
    4+: Emergency retirement — close all, post-mortem within 48h
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE LEVELS
# ─────────────────────────────────────────────────────────────────────────────
class DecayResponse(Enum):
    NORMAL            = "NORMAL"
    MONITOR_INCREASED = "MONITOR_INCREASED"      # 1 condition
    REDUCE_50         = "REDUCE_50_PCT"          # 2 conditions
    SUSPEND           = "SUSPEND_PENDING_REVIEW" # 3 conditions
    EMERGENCY_RETIRE  = "EMERGENCY_RETIRE"       # 4+ conditions


# ─────────────────────────────────────────────────────────────────────────────
# DECAY METRICS SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DecayMetrics:
    """Point-in-time decay metrics for one strategy."""
    strategy: str
    instrument: str

    # Rolling performance
    rolling_sharpe_30d: float
    historical_sharpe: float
    sharpe_ratio: float             # rolling / historical

    # IC metrics
    rolling_ic_30d: float
    ic_declining: bool              # 30d mean < 70% of prior 30d mean
    consecutive_negative_ic_days: int

    # Execution quality
    slippage_ratio: float           # recent / baseline
    slippage_alert: bool

    # CUSUM
    cusum_value: float
    cusum_threshold: float
    cusum_alert: bool

    # Turnover stability
    turnover_cv: float              # Coefficient of variation of daily trade count

    # Condition flags
    condition_a: bool               # Sharpe < 50% historical
    condition_b: bool               # IC < 0 for ≥ 10 days
    condition_c: bool               # Hostile regime > 20 days
    condition_d: bool               # Slippage > 20% baseline
    condition_e: bool               # CUSUM alert

    n_conditions: int
    response: DecayResponse
    active_reasons: list[str]

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def log_summary(self) -> str:
        return (
            f"{self.strategy}/{self.instrument} | "
            f"response={self.response.value} | "
            f"conditions={self.n_conditions} | "
            f"SR30d={self.rolling_sharpe_30d:.2f} | "
            f"IC30d={self.rolling_ic_30d:.4f} | "
            f"slip_ratio={self.slippage_ratio:.2f} | "
            f"cusum={'ALERT' if self.cusum_alert else 'ok'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CUSUM TEST
# ─────────────────────────────────────────────────────────────────────────────
def cusum_test(
    daily_pnl: np.ndarray,
    null_window: int = 60,
    sigma_multiplier: float = 3.0,
) -> dict:
    """
    CUSUM test for structural break in P&L series.
    Calibrated at 5% false-positive rate.

    S_t = S_{t-1} + (r_t - μ_null) / σ_r
    Alert: |S_T| > sigma_multiplier × sqrt(T)

    Uses first null_window observations to estimate null distribution
    (strategy is assumed to work during its initial deployment phase).
    """
    n = len(daily_pnl)
    if n < null_window + 5:
        return {
            "alert": False,
            "cusum_value": 0.0,
            "threshold": float("nan"),
            "reason": f"insufficient_data({n} < {null_window + 5})",
        }

    null_pnl = daily_pnl[:null_window]
    mu_null = float(np.mean(null_pnl))
    sigma_r = float(np.std(null_pnl, ddof=1))

    if sigma_r < 1e-10:
        return {
            "alert": False,
            "cusum_value": 0.0,
            "threshold": float("nan"),
            "reason": "zero_variance_in_null_window",
        }

    cusum = np.cumsum((daily_pnl - mu_null) / sigma_r)
    cusum_stat = float(abs(cusum[-1]))
    threshold = sigma_multiplier * np.sqrt(n)
    alert = cusum_stat > threshold

    if alert:
        logger.warning(
            f"CUSUM alert: stat={cusum_stat:.2f} > threshold={threshold:.2f} "
            f"(n={n}, sigma_mult={sigma_multiplier})"
        )

    return {
        "alert": alert,
        "cusum_value": float(cusum_stat),
        "threshold": float(threshold),
        "n_observations": n,
        "cusum_series": cusum[-30:].tolist(),  # Last 30 values for charting
    }


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA DECAY MONITOR
# ─────────────────────────────────────────────────────────────────────────────
class AlphaDecayMonitor:
    """
    Monitors each active strategy/instrument combination for alpha decay.
    Runs daily and on-demand.

    Maintains rolling state:
        - Daily P&L history per strategy
        - Signal strength (IC proxy) per strategy
        - Slippage records (from fill tracker)
        - Consecutive negative-IC day counter
        - Hostile regime day counter
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        d = config.decay_params

        # Thresholds from risk_limits.yaml
        self._sharpe_warn = float(d.get("rolling_sharpe_warn_threshold", 0.50))
        self._ic_neg_days = int(d.get("rolling_ic_negative_days", 10))
        self._hostile_days = int(d.get("hostile_regime_days", 20))
        self._slip_pct = float(d.get("slippage_deterioration_pct", 0.20))
        self._cusum_sigma = float(d.get("cusum_alert_sigma", 3.0))

        # Per-strategy state
        self._daily_pnl: dict[str, list[float]] = {}
        self._signal_strengths: dict[str, list[float]] = {}
        self._slippage_history: dict[str, list[float]] = {}
        self._neg_ic_counter: dict[str, int] = {}
        self._hostile_counter: dict[str, int] = {}
        self._historical_sharpe: dict[str, float] = {}
        self._alert_history: list[DecayMetrics] = []

    # ── Data ingestion ────────────────────────────────────────────────────────
    def record_daily_pnl(
        self,
        strategy: str,
        instrument: str,
        pnl: float,
    ) -> None:
        """Record end-of-day P&L for a strategy/instrument."""
        key = f"{strategy}::{instrument}"
        if key not in self._daily_pnl:
            self._daily_pnl[key] = []
        self._daily_pnl[key].append(pnl)

    def record_signal_strength(
        self,
        strategy: str,
        instrument: str,
        signal_strength: float,
        trade_pnl: Optional[float] = None,
    ) -> None:
        """
        Record signal strength for IC computation.
        signal_strength: normalised signal magnitude [0,1] at trade entry.
        trade_pnl: if provided, used for realised IC computation.
        """
        key = f"{strategy}::{instrument}"
        if key not in self._signal_strengths:
            self._signal_strengths[key] = []
        self._signal_strengths[key].append(signal_strength)

    def record_slippage(
        self,
        strategy: str,
        instrument: str,
        slippage_pts: float,
    ) -> None:
        """Record a slippage observation for execution quality monitoring."""
        key = f"{strategy}::{instrument}"
        if key not in self._slippage_history:
            self._slippage_history[key] = []
        self._slippage_history[key].append(abs(slippage_pts))

    def update_hostile_regime(
        self,
        strategy: str,
        instrument: str,
        is_hostile: bool,
    ) -> None:
        """
        Update consecutive hostile-regime counter.
        Hostile regime: strategy type operating in adverse regime
        (e.g. mean reversion active in strongly trending market).
        """
        key = f"{strategy}::{instrument}"
        if is_hostile:
            self._hostile_counter[key] = self._hostile_counter.get(key, 0) + 1
        else:
            self._hostile_counter[key] = 0

    def set_historical_sharpe(
        self,
        strategy: str,
        instrument: str,
        sharpe: float,
    ) -> None:
        """Set the validated historical Sharpe for a strategy (from paper trading)."""
        key = f"{strategy}::{instrument}"
        self._historical_sharpe[key] = sharpe

    # ── Main compute ──────────────────────────────────────────────────────────
    def compute_metrics(
        self,
        strategy: str,
        instrument: str,
    ) -> DecayMetrics:
        """
        Compute full decay metrics for a strategy/instrument pair.
        Called daily by the main orchestrator.
        """
        key = f"{strategy}::{instrument}"

        pnl_list = self._daily_pnl.get(key, [])
        signal_list = self._signal_strengths.get(key, [])
        slippage_list = self._slippage_history.get(key, [])
        historical_sharpe = self._historical_sharpe.get(key, 1.0)
        hostile_days = self._hostile_counter.get(key, 0)

        daily_pnl = np.array(pnl_list) if pnl_list else np.array([0.0])

        # ── Rolling 30-day Sharpe (Condition A) ───────────────────────────────
        rolling_sr = self._rolling_sharpe(daily_pnl, window=30)
        sharpe_ratio = (
            rolling_sr / historical_sharpe
            if historical_sharpe > 1e-10 else 0.0
        )
        condition_a = rolling_sr < self._sharpe_warn * historical_sharpe

        # ── Rolling IC (Condition B) ───────────────────────────────────────────
        rolling_ic = self._rolling_ic(daily_pnl, signal_list, window=30)
        ic_prev = self._rolling_ic(daily_pnl[:-30] if len(daily_pnl) > 30 else daily_pnl,
                                    signal_list, window=30)
        ic_declining = (
            rolling_ic < ic_prev * 0.70
            if abs(ic_prev) > 1e-10 else False
        )

        # Update consecutive negative-IC counter
        if rolling_ic < 0:
            self._neg_ic_counter[key] = self._neg_ic_counter.get(key, 0) + 1
        else:
            self._neg_ic_counter[key] = 0

        consecutive_neg = self._neg_ic_counter.get(key, 0)
        condition_b = consecutive_neg >= self._ic_neg_days

        # ── Hostile regime (Condition C) ──────────────────────────────────────
        condition_c = hostile_days >= self._hostile_days

        # ── Slippage (Condition D) ─────────────────────────────────────────────
        slip_ratio = self._slippage_ratio(slippage_list)
        condition_d = slip_ratio > (1 + self._slip_pct)

        # ── CUSUM (Condition E) ────────────────────────────────────────────────
        cusum_result = cusum_test(
            daily_pnl,
            null_window=min(60, len(daily_pnl) // 2),
            sigma_multiplier=self._cusum_sigma,
        )
        condition_e = cusum_result["alert"]

        # ── Turnover stability ────────────────────────────────────────────────
        turnover_cv = self._turnover_cv(pnl_list)

        # ── Count active conditions and determine response ─────────────────────
        conditions = {
            "A_low_sharpe": condition_a,
            "B_negative_ic": condition_b,
            "C_hostile_regime": condition_c,
            "D_slippage": condition_d,
            "E_cusum": condition_e,
        }
        active = [name for name, active in conditions.items() if active]
        n_active = len(active)

        # Weight: Sharpe and CUSUM are weighted 2x (more serious)
        weighted = sum([
            2 if condition_a else 0,
            2 if condition_b else 0,
            1 if condition_c else 0,
            1 if condition_d else 0,
            2 if condition_e else 0,
        ])

        response = self._determine_response(weighted, n_active)

        metrics = DecayMetrics(
            strategy=strategy,
            instrument=instrument,
            rolling_sharpe_30d=float(rolling_sr),
            historical_sharpe=float(historical_sharpe),
            sharpe_ratio=float(sharpe_ratio),
            rolling_ic_30d=float(rolling_ic),
            ic_declining=bool(ic_declining),
            consecutive_negative_ic_days=int(consecutive_neg),
            slippage_ratio=float(slip_ratio),
            slippage_alert=bool(condition_d),
            cusum_value=float(cusum_result["cusum_value"]),
            cusum_threshold=float(cusum_result.get("threshold", float("nan"))),
            cusum_alert=bool(condition_e),
            turnover_cv=float(turnover_cv),
            condition_a=condition_a,
            condition_b=condition_b,
            condition_c=condition_c,
            condition_d=condition_d,
            condition_e=condition_e,
            n_conditions=n_active,
            response=response,
            active_reasons=active,
        )

        if n_active > 0:
            logger.warning(f"Decay alert: {metrics.log_summary()}")
            self._alert_history.append(metrics)
        else:
            logger.debug(f"Decay check: {metrics.log_summary()}")

        return metrics

    def compute_all(
        self,
        strategy_instrument_pairs: list[tuple[str, str]],
    ) -> dict[str, DecayMetrics]:
        """
        Compute decay metrics for all active strategy/instrument pairs.
        Returns {key: DecayMetrics}.
        """
        results = {}
        for strategy, instrument in strategy_instrument_pairs:
            key = f"{strategy}::{instrument}"
            results[key] = self.compute_metrics(strategy, instrument)
        return results

    def get_allocation_scale(self, metrics: DecayMetrics) -> float:
        """
        Return position size scale factor based on decay response.
        Used by risk engine to reduce sizing when decay detected.
        """
        d = self.config.decay_params
        scales = {
            DecayResponse.NORMAL:            1.00,
            DecayResponse.MONITOR_INCREASED: float(d.get("response_1_condition_allocation_pct", 0.75)),
            DecayResponse.REDUCE_50:         float(d.get("response_2_condition_allocation_pct", 0.50)),
            DecayResponse.SUSPEND:           0.00,
            DecayResponse.EMERGENCY_RETIRE:  0.00,
        }
        return scales.get(metrics.response, 1.0)

    # ── Indicator helpers ─────────────────────────────────────────────────────
    def _rolling_sharpe(self, pnl: np.ndarray, window: int = 30) -> float:
        if len(pnl) < window // 2:
            return 0.0
        arr = pnl[-window:]
        std = float(np.std(arr, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(252))

    def _rolling_ic(
        self,
        pnl: np.ndarray,
        signals: list[float],
        window: int = 30,
    ) -> float:
        """
        Rolling Information Coefficient: correlation(signal_strength, pnl).
        Proxy for whether the signal is still predicting direction.
        """
        n = min(len(pnl), len(signals), window)
        if n < 5:
            return 0.0

        pnl_w = pnl[-n:]
        sig_w = np.array(signals[-n:])

        if np.std(sig_w) < 1e-10 or np.std(pnl_w) < 1e-10:
            return 0.0

        corr = float(np.corrcoef(sig_w, pnl_w)[0, 1])
        return corr if not np.isnan(corr) else 0.0

    def _slippage_ratio(self, slippage_history: list[float]) -> float:
        """recent_avg / baseline_avg — 1.0 means no deterioration."""
        if len(slippage_history) < 100:
            return 1.0
        baseline = float(np.mean(slippage_history[:100]))
        recent = float(np.mean(slippage_history[-50:]))
        if baseline < 1e-10:
            return 1.0
        return recent / baseline

    def _turnover_cv(self, pnl_list: list[float], window: int = 30) -> float:
        """
        Coefficient of variation of daily trade count.
        High CV = unstable signal generation = possible decay indicator.
        Uses daily P&L sign changes as a trade-count proxy.
        """
        if len(pnl_list) < window:
            return 0.0
        pnl = np.array(pnl_list[-window:])
        # Count non-zero days as proxy for active trading days
        active = np.abs(pnl) > 1e-10
        if active.sum() < 5:
            return 0.0
        # Use rolling 5-day sums as "weekly activity"
        activity = np.array([active[i:i+5].sum() for i in range(0, len(active)-4, 5)])
        if len(activity) < 2 or np.mean(activity) < 1e-10:
            return 0.0
        return float(np.std(activity) / np.mean(activity))

    def _determine_response(self, weighted_score: int, n_conditions: int) -> DecayResponse:
        """Map weighted condition score to response level."""
        d = self.config.decay_params
        if n_conditions == 0:
            return DecayResponse.NORMAL
        if d.get("response_4plus_emergency_retire") and weighted_score >= 6:
            return DecayResponse.EMERGENCY_RETIRE
        if d.get("response_3_condition_suspend") and n_conditions >= 3:
            return DecayResponse.SUSPEND
        if n_conditions >= 2:
            return DecayResponse.REDUCE_50
        return DecayResponse.MONITOR_INCREASED

    # ── History ───────────────────────────────────────────────────────────────
    @property
    def alert_history(self) -> list[DecayMetrics]:
        return list(self._alert_history)

    def recent_alerts(self, n: int = 20) -> list[DecayMetrics]:
        return self._alert_history[-n:]
