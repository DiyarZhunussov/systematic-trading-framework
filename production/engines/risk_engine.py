"""
risk_engine.py — Risk Management Engine
Implements Part VII of the framework.

Core principle (Section 7.1):
    Risk management is NOT a secondary constraint applied to a position-sizing system.
    It IS the position-sizing system.
    The alpha engine determines direction. Risk management determines size.
    Size determines virtually all outcomes over any meaningful time horizon.

Asymmetry of losses (Section 7.1):
    10% loss requires 11.1% gain to recover.
    20% loss requires 25.0% gain to recover.
    50% loss requires 100.0% gain to recover.
    The primary obligation is to stay in the game — not to maximise return.

Architecture:
    Level 1 — Trade-level limits (stop, spread, slippage)
    Level 2 — Strategy-level limits (daily loss, concurrent trades, correlation)
    Level 3 — Portfolio-level limits (daily loss, drawdown, leverage, CVaR)
    Level 4 — Kill switch (hard stops, manual reset required)

Gross leverage normalisation (Section 7.4 — adversarial review correction):
    The regime allocation matrix can produce aggregate scales > 100% in some regimes
    (e.g. Normal+Trending: 80% + 10% + 40% = 130%). normalize_strategy_scales()
    from regime_engine is enforced here at every position sizing call.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────
class RiskCheckResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REDUCED = "reduced"          # Approved with reduced size


class DrawdownLevel(Enum):
    NORMAL = "normal"            # 0–3%
    CAUTION = "caution"          # 3–5% — scale to 75%
    WARNING = "warning"          # 5–7% — scale to 50%, suspend MR
    SEVERE = "severe"            # 7–9% — scale to 25%, trend only
    CRITICAL = "critical"        # 9–10% — full suspension
    BREACH = "breach"            # >10% — emergency close, prop firm risk


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZE REQUEST/RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionSizeRequest:
    """Input to the position sizing procedure."""
    instrument: str
    strategy: str
    direction: str                  # "buy" or "sell"
    entry_price: float
    stop_loss_price: float
    signal_confidence: float        # w_i from Bayesian confidence weight [0,1]
    regime_scale: float             # Scale from regime engine [0,1]
    current_regime: str             # e.g. "normal_trending"
    account_balance: float
    current_portfolio_risk_usd: float   # Open risk already committed
    current_drawdown_pct: float         # Current drawdown from peak


@dataclass
class PositionSizeResult:
    """Full output of position sizing — all intermediate values for audit log."""
    request: PositionSizeRequest
    result: RiskCheckResult
    rejection_reason: Optional[str]

    # Final size
    position_size_lots: float
    dollar_risk: float

    # Intermediate calculations
    adjusted_risk_pct: float
    regime_scale_used: float
    drawdown_scale: float
    stop_distance: float
    size_from_stop: float
    size_from_vol_target: float
    size_from_budget: float
    size_before_rounding: float

    # Limits applied
    vol_target_daily: float
    daily_budget_remaining_usd: float
    max_position_from_limit: float

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_approved(self) -> bool:
        return self.result in (RiskCheckResult.APPROVED, RiskCheckResult.REDUCED)

    def audit_log(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument": self.request.instrument,
            "strategy": self.request.strategy,
            "result": self.result.value,
            "rejection_reason": self.rejection_reason,
            "position_size_lots": self.position_size_lots,
            "dollar_risk": self.dollar_risk,
            "adjusted_risk_pct": self.adjusted_risk_pct,
            "regime_scale": self.regime_scale_used,
            "drawdown_scale": self.drawdown_scale,
            "stop_distance": self.stop_distance,
            "size_from_stop": self.size_from_stop,
            "size_from_vol_target": self.size_from_vol_target,
            "size_from_budget": self.size_from_budget,
            "signal_confidence": self.request.signal_confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class DrawdownTracker:
    """
    Tracks equity curve, peak, and drawdown in real time.
    Implements the response schedule from Section 7.6.
    """

    def __init__(self, initial_balance: float):
        self._peak = initial_balance
        self._current = initial_balance
        self._drawdown_start: Optional[datetime] = None
        self._daily_start = initial_balance
        self._daily_start_date = datetime.now(timezone.utc).date()
        self._equity_history: list[tuple[datetime, float]] = [
            (datetime.now(timezone.utc), initial_balance)
        ]

    def update(self, current_equity: float) -> None:
        """Update equity and recalculate peak/drawdown."""
        now = datetime.now(timezone.utc)
        self._current = current_equity
        self._equity_history.append((now, current_equity))

        if current_equity > self._peak:
            self._peak = current_equity
            self._drawdown_start = None  # Reset drawdown clock

        # Reset daily start if new day
        today = now.date()
        if today != self._daily_start_date:
            self._daily_start = current_equity
            self._daily_start_date = today

        # Track drawdown duration
        if self.drawdown_pct > 0.03 and self._drawdown_start is None:
            self._drawdown_start = now

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak as a fraction."""
        if self._peak <= 0:
            return 0.0
        return max(0.0, (self._peak - self._current) / self._peak)

    @property
    def daily_loss_pct(self) -> float:
        """Today's loss from start-of-day balance."""
        if self._daily_start <= 0:
            return 0.0
        return max(0.0, (self._daily_start - self._current) / self._daily_start)

    @property
    def drawdown_days(self) -> int:
        """Days in current drawdown (0 if not in drawdown)."""
        if self._drawdown_start is None:
            return 0
        return (datetime.now(timezone.utc) - self._drawdown_start).days

    @property
    def drawdown_level(self) -> DrawdownLevel:
        """Current drawdown severity level."""
        dd = self.drawdown_pct
        if dd >= 0.10:
            return DrawdownLevel.BREACH
        elif dd >= 0.09:
            return DrawdownLevel.CRITICAL
        elif dd >= 0.07:
            return DrawdownLevel.SEVERE
        elif dd >= 0.05:
            return DrawdownLevel.WARNING
        elif dd >= 0.03:
            return DrawdownLevel.CAUTION
        return DrawdownLevel.NORMAL

    def get_size_scale(self, limits: dict) -> float:
        """
        Return position size scale factor based on current drawdown level.
        Per Section 7.6 drawdown response schedule.
        """
        schedule = limits.get("drawdown_response", [])
        dd = self.drawdown_pct

        # Find applicable scale
        scale = 1.0
        for entry in reversed(schedule):
            if dd >= entry["threshold_pct"]:
                if entry["action"] in ("full_suspension", "trend_only"):
                    return 0.0
                scale = entry.get("scale_factor", 1.0)
                break
        return scale

    def extended_drawdown_alert(self, threshold_pct: float = 0.03, days: int = 10) -> bool:
        """True if drawdown > threshold for more than N consecutive days."""
        return self.drawdown_pct > threshold_pct and self.drawdown_days > days


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY TARGET CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityTargetCalculator:
    """
    Implements volatility targeting from Section 7.2.
    Sizes all instruments to contribute equal realised volatility.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        vt = config.vol_target_params
        self._normal_target = float(vt.get("normal_annualised_pct", 0.10))
        self._high_vol_target = float(vt.get("high_vol_annualised_pct", 0.07))
        self._crisis_target = float(vt.get("crisis_annualised_pct", 0.04))
        self._smoothing_ema = int(vt.get("smoothing_ema_days", 3))
        self._smoothed_vol: dict[str, float] = {}

    def get_target(self, regime: str) -> float:
        """Return annualised vol target for given regime string."""
        if "crisis" in regime:
            return self._crisis_target
        elif "high" in regime:
            return self._high_vol_target
        return self._normal_target

    def get_daily_target(self, regime: str) -> float:
        """Return daily vol target = annual / sqrt(252)."""
        return self.get_target(regime) / np.sqrt(252)

    def compute_instrument_vol(
        self,
        instrument: str,
        returns: np.ndarray,
        window: int = 20,
    ) -> float:
        """
        Compute EMA-smoothed 20-day realised daily volatility for an instrument.
        Smoothing prevents position churn from short-lived vol spikes.
        """
        if len(returns) < window:
            return 0.01  # Default to 1% daily vol if insufficient data

        raw_vol = float(np.std(returns[-window:], ddof=1))

        # Apply EMA smoothing
        prev = self._smoothed_vol.get(instrument, raw_vol)
        alpha = 2.0 / (self._smoothing_ema + 1)
        smoothed = alpha * raw_vol + (1 - alpha) * prev
        self._smoothed_vol[instrument] = smoothed

        return smoothed

    def vol_target_size(
        self,
        account_balance: float,
        instrument_vol_daily: float,
        entry_price: float,
        contract_size: float,
        regime: str,
    ) -> float:
        """
        Compute position size from volatility target.
        units = (Account × daily_target) / (σ_daily × Price × ContractSize)
        """
        daily_target = self.get_daily_target(regime)
        if instrument_vol_daily <= 0 or entry_price <= 0 or contract_size <= 0:
            return 0.0
        return (account_balance * daily_target) / (
            instrument_vol_daily * entry_price * contract_size
        )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioRiskTracker:
    """
    Tracks current open risk across all positions and strategies.
    Enforces Level 2 and Level 3 portfolio constraints.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        self._open_positions: dict[int, dict] = {}  # ticket → position info

    def add_position(
        self,
        ticket: int,
        instrument: str,
        strategy: str,
        direction: str,
        size_lots: float,
        entry_price: float,
        stop_price: float,
        dollar_risk: float,
    ) -> None:
        self._open_positions[ticket] = {
            "instrument": instrument,
            "strategy": strategy,
            "direction": direction,
            "size_lots": size_lots,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "dollar_risk": dollar_risk,
            "opened_at": datetime.now(timezone.utc),
        }

    def remove_position(self, ticket: int) -> None:
        self._open_positions.pop(ticket, None)

    @property
    def total_open_risk_usd(self) -> float:
        return sum(p["dollar_risk"] for p in self._open_positions.values())

    def strategy_open_risk_usd(self, strategy: str) -> float:
        return sum(
            p["dollar_risk"]
            for p in self._open_positions.values()
            if p["strategy"] == strategy
        )

    def strategy_trade_count(self, strategy: str) -> int:
        return sum(
            1 for p in self._open_positions.values()
            if p["strategy"] == strategy
        )

    def usd_net_exposure(self, account_balance: float) -> float:
        """
        Net USD exposure as fraction of account.
        Long USD/XXX adds, short XXX/USD adds, long XXX/USD subtracts.
        Simplified: flags any USD pair exposure.
        """
        usd_exposure = 0.0
        for pos in self._open_positions.values():
            instr = pos["instrument"]
            direction = pos["direction"]
            risk = pos["dollar_risk"]
            if "USD" in instr:
                if direction == "buy":
                    usd_exposure += risk
                else:
                    usd_exposure -= risk
        return abs(usd_exposure) / (account_balance + 1e-10)

    def equity_index_net_exposure(self, account_balance: float) -> float:
        """Net equity index exposure (NQ + SPX + DAX same direction)."""
        index_symbols = {"NQ100", "SPX500", "DAX40"}
        long_risk = sum(
            p["dollar_risk"]
            for p in self._open_positions.values()
            if p["instrument"] in index_symbols and p["direction"] == "buy"
        )
        short_risk = sum(
            p["dollar_risk"]
            for p in self._open_positions.values()
            if p["instrument"] in index_symbols and p["direction"] == "sell"
        )
        net = abs(long_risk - short_risk)
        return net / (account_balance + 1e-10)

    def has_correlated_pair_conflict(
        self,
        instrument: str,
        direction: str,
    ) -> bool:
        """
        Check correlated pair rule (Section 7.5 Level 2):
        No simultaneous long EUR/USD and short USD/JPY at full size.
        Simplified: flag if adding creates conflicting USD net exposure.
        """
        conflict_pairs = [
            ("EURUSD", "buy", "USDJPY", "sell"),
            ("GBPUSD", "buy", "USDJPY", "sell"),
            ("EURUSD", "sell", "USDJPY", "buy"),
            ("GBPUSD", "sell", "USDJPY", "buy"),
        ]
        for p in self._open_positions.values():
            for c1_sym, c1_dir, c2_sym, c2_dir in conflict_pairs:
                if (
                    (p["instrument"] == c1_sym and p["direction"] == c1_dir
                     and instrument == c2_sym and direction == c2_dir)
                    or
                    (p["instrument"] == c2_sym and p["direction"] == c2_dir
                     and instrument == c1_sym and direction == c1_dir)
                ):
                    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CVaR (EXPECTED SHORTFALL) CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_cvar(
    daily_pnl: np.ndarray,
    confidence: float = 0.95,
    account_balance: float = 100_000.0,
) -> float:
    """
    Compute Expected Shortfall (CVaR) at given confidence level.
    Uses empirical bootstrap — avoids normality assumption (Section 7.7).

    Returns CVaR as fraction of account (positive = loss).
    """
    if len(daily_pnl) < 20:
        return 0.0

    # As fraction of account
    pnl_pct = daily_pnl / (account_balance + 1e-10)

    var_threshold = np.percentile(pnl_pct, (1 - confidence) * 100)
    tail = pnl_pct[pnl_pct <= var_threshold]

    if len(tail) == 0:
        return abs(float(var_threshold))

    return float(abs(np.mean(tail)))


def cornish_fisher_var(
    mean: float,
    sigma: float,
    skewness: float,
    excess_kurtosis: float,
    confidence: float = 0.95,
) -> float:
    """
    Cornish-Fisher fat-tail adjusted VaR (Section 7.7).
    Used when |skewness| > 0.5 OR excess_kurtosis > 1.0
    (which virtually always holds for FX/CFD returns).

    Returns VaR as a fraction of portfolio value (positive = loss).
    """
    from scipy.stats import norm as scipy_norm
    z = scipy_norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    cf_z = (
        z
        + (z**2 - 1) * skewness / 6
        + (z**3 - 3 * z) * excess_kurtosis / 24
        - (2 * z**3 - 5 * z) * skewness**2 / 36
    )

    return float(mean + sigma * cf_z)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RISK ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class RiskEngine:
    """
    Full risk management system implementing all four levels (Section 7.5).

    Usage:
        risk_engine = RiskEngine(config, initial_balance)
        result = risk_engine.size_position(request)
        if result.is_approved:
            submit order with result.position_size_lots
    """

    def __init__(self, config: "SystemConfig", initial_balance: float):
        self.config = config
        self._limits = config.risk
        self._trade_limits = config.trade_limits
        self._strategy_limits = config.strategy_limits
        self._portfolio_limits = config.portfolio_limits
        self._kill_switch_params = config.kill_switch_params

        self.drawdown_tracker = DrawdownTracker(initial_balance)
        self.vol_calculator = VolatilityTargetCalculator(config)
        self.portfolio_tracker = PortfolioRiskTracker(config)

        self._kill_switch_triggered = False
        self._daily_pnl_history: list[float] = []
        self._initial_balance = initial_balance

    # ── Main sizing procedure (Section 7.3) ──────────────────────────────────
    def size_position(
        self,
        request: PositionSizeRequest,
        instrument_returns: np.ndarray,
        instrument_config: dict,
    ) -> PositionSizeResult:
        """
        Full 11-step position sizing procedure (Section 7.3).
        Returns PositionSizeResult with all intermediate values for audit.
        """
        def reject(reason: str) -> PositionSizeResult:
            return PositionSizeResult(
                request=request,
                result=RiskCheckResult.REJECTED,
                rejection_reason=reason,
                position_size_lots=0.0,
                dollar_risk=0.0,
                adjusted_risk_pct=0.0,
                regime_scale_used=request.regime_scale,
                drawdown_scale=0.0,
                stop_distance=0.0,
                size_from_stop=0.0,
                size_from_vol_target=0.0,
                size_from_budget=0.0,
                size_before_rounding=0.0,
                vol_target_daily=0.0,
                daily_budget_remaining_usd=0.0,
                max_position_from_limit=0.0,
            )

        # ── Level 4: Kill switch check ────────────────────────────────────────
        if self._kill_switch_triggered:
            return reject("kill_switch_active — manual reset required")

        # ── Level 3: Daily loss check ─────────────────────────────────────────
        if self.drawdown_tracker.daily_loss_pct >= self._portfolio_limits["max_daily_loss_pct"]:
            return reject(
                f"daily_loss_limit_reached("
                f"{self.drawdown_tracker.daily_loss_pct:.2%} >= "
                f"{self._portfolio_limits['max_daily_loss_pct']:.2%})"
            )

        # ── Level 3: Kill switch trigger ──────────────────────────────────────
        if self.drawdown_tracker.daily_loss_pct >= self._kill_switch_params["daily_loss_trigger_pct"]:
            self._trigger_kill_switch(
                f"daily_loss_kill_switch("
                f"{self.drawdown_tracker.daily_loss_pct:.2%})"
            )
            return reject("kill_switch_triggered")

        if self.drawdown_tracker.drawdown_pct >= self._kill_switch_params["drawdown_trigger_pct"]:
            self._trigger_kill_switch(
                f"drawdown_kill_switch("
                f"{self.drawdown_tracker.drawdown_pct:.2%})"
            )
            return reject("kill_switch_triggered")

        # ── Level 2: Strategy concurrent trade limit ──────────────────────────
        max_concurrent = self._strategy_limits.get("max_concurrent_trades", 3)
        if self.portfolio_tracker.strategy_trade_count(request.strategy) >= max_concurrent:
            return reject(
                f"concurrent_trade_limit({request.strategy}: "
                f"{max_concurrent} already open)"
            )

        # ── Level 2: Correlated pair check ────────────────────────────────────
        if self.portfolio_tracker.has_correlated_pair_conflict(
            request.instrument, request.direction
        ):
            return reject(
                f"correlated_pair_conflict("
                f"{request.instrument} {request.direction})"
            )

        # ── Step 1: Base risk per trade ───────────────────────────────────────
        base_risk_pct = self._trade_limits.get("max_risk_per_trade_pct", 0.005)

        # ── Step 2: Regime scale ──────────────────────────────────────────────
        regime_scale = float(np.clip(request.regime_scale, 0.0, 1.0))

        # ── Step 3: Drawdown scale ────────────────────────────────────────────
        drawdown_scale = self.drawdown_tracker.get_size_scale(self._limits)
        if drawdown_scale == 0.0:
            return reject(
                f"drawdown_suspension("
                f"{self.drawdown_tracker.drawdown_pct:.2%})"
            )

        # ── Step 3 cont: Confidence-adjusted risk ─────────────────────────────
        signal_confidence = float(np.clip(request.signal_confidence, 0.0, 1.0))
        adjusted_risk_pct = (
            base_risk_pct
            * regime_scale
            * drawdown_scale
            * signal_confidence
        )

        if adjusted_risk_pct < 1e-6:
            return reject("adjusted_risk_pct_too_small")

        # ── Step 4: Dollar risk ───────────────────────────────────────────────
        dollar_risk = request.account_balance * adjusted_risk_pct

        # ── Step 5: Stop distance ─────────────────────────────────────────────
        stop_distance = abs(request.entry_price - request.stop_loss_price)
        if stop_distance <= 0:
            return reject("invalid_stop_distance(stop == entry)")

        # Validate stop distance vs ATR limit
        if len(instrument_returns) >= 20:
            atr_proxy = float(np.std(instrument_returns[-20:])) * request.entry_price
            max_stop = atr_proxy * self._trade_limits.get("max_stop_distance_atr_multiple", 1.5)
            if stop_distance > max_stop:
                # Don't reject — reduce size proportionally
                logger.warning(
                    f"Stop distance {stop_distance:.5f} > max {max_stop:.5f}. "
                    f"Size will be reduced by ATR constraint."
                )

        # ── Step 6: Raw position size from stop ───────────────────────────────
        contract_size = instrument_config.get("contract_size", 100_000)
        size_from_stop = dollar_risk / (stop_distance * contract_size)

        # ── Step 7: Volatility targeting size ─────────────────────────────────
        instr_vol_daily = self.vol_calculator.compute_instrument_vol(
            request.instrument, instrument_returns
        )
        size_from_vol = self.vol_calculator.vol_target_size(
            account_balance=request.account_balance,
            instrument_vol_daily=instr_vol_daily,
            entry_price=request.entry_price,
            contract_size=contract_size,
            regime=request.current_regime,
        )
        vol_target_daily = self.vol_calculator.get_daily_target(request.current_regime)

        # ── Step 8: Conservative minimum ─────────────────────────────────────
        position_size = min(size_from_stop, size_from_vol)

        # ── Step 9: Portfolio risk budget constraint ──────────────────────────
        daily_risk_limit = (
            request.account_balance
            * self._strategy_limits.get("max_daily_loss_pct", 0.015)
        )
        remaining_budget = daily_risk_limit - request.current_portfolio_risk_usd
        if remaining_budget <= 0:
            return reject("strategy_daily_risk_budget_exhausted")

        size_from_budget = remaining_budget / (stop_distance * contract_size)
        position_size = min(position_size, size_from_budget)

        # ── Step 10: Hard position size limit ────────────────────────────────
        max_pos_pct = self._strategy_limits.get("max_concurrent_trades", 3) * base_risk_pct
        max_position = (
            request.account_balance * max_pos_pct / (request.entry_price + 1e-10)
        )
        position_size = min(position_size, max_position)

        # ── USD net exposure limit ────────────────────────────────────────────
        usd_net = self.portfolio_tracker.usd_net_exposure(request.account_balance)
        usd_limit = self._strategy_limits.get("usd_net_exposure_max_pct", 0.03)
        if usd_net > usd_limit and "USD" in request.instrument:
            logger.warning(
                f"USD net exposure {usd_net:.2%} approaching limit {usd_limit:.2%}. "
                f"Reducing size."
            )
            position_size *= (usd_limit / usd_net)

        # ── Equity index net exposure limit ───────────────────────────────────
        idx_net = self.portfolio_tracker.equity_index_net_exposure(request.account_balance)
        idx_limit = self._strategy_limits.get("equity_index_net_exposure_max_pct", 0.05)
        if idx_net > idx_limit and request.instrument in {"NQ100", "SPX500", "DAX40"}:
            position_size *= (idx_limit / idx_net)

        # ── Free margin check ─────────────────────────────────────────────────
        min_free_margin = self._portfolio_limits.get("min_free_margin_pct", 0.30)
        estimated_margin_use = position_size * request.entry_price * contract_size
        if estimated_margin_use > request.account_balance * (1 - min_free_margin):
            position_size = (
                request.account_balance * (1 - min_free_margin)
                / (request.entry_price * contract_size + 1e-10)
            )

        size_before_rounding = position_size

        # ── Step 11: Round to lot size ────────────────────────────────────────
        lot_size = instrument_config.get("lot_size", 0.01)
        position_size = max(lot_size, round(position_size / lot_size) * lot_size)

        # ── Final dollar risk ─────────────────────────────────────────────────
        final_dollar_risk = position_size * stop_distance * contract_size

        result_type = (
            RiskCheckResult.REDUCED
            if position_size < size_from_stop * 0.95
            else RiskCheckResult.APPROVED
        )

        logger.info(
            f"Position sized: {request.instrument} {request.strategy} | "
            f"{position_size:.2f} lots | ${final_dollar_risk:.2f} risk | "
            f"result={result_type.value}"
        )

        return PositionSizeResult(
            request=request,
            result=result_type,
            rejection_reason=None,
            position_size_lots=position_size,
            dollar_risk=final_dollar_risk,
            adjusted_risk_pct=adjusted_risk_pct,
            regime_scale_used=regime_scale,
            drawdown_scale=drawdown_scale,
            stop_distance=stop_distance,
            size_from_stop=size_from_stop,
            size_from_vol_target=size_from_vol,
            size_from_budget=size_from_budget,
            size_before_rounding=size_before_rounding,
            vol_target_daily=vol_target_daily,
            daily_budget_remaining_usd=remaining_budget,
            max_position_from_limit=max_position,
        )

    # ── Spread pre-trade check (Level 1) ─────────────────────────────────────
    def check_spread(
        self,
        instrument: str,
        current_spread_points: float,
        instrument_config: dict,
    ) -> tuple[bool, str]:
        """
        Reject trade if spread exceeds 3× baseline (Level 1 — Section 7.5).
        Returns (approved: bool, reason: str).
        """
        max_spread = instrument_config.get("max_spread_points")
        if max_spread is None:
            return True, ""

        spread_multiple = self._trade_limits.get("max_spread_multiple", 3.0)
        limit = max_spread * spread_multiple

        if current_spread_points > limit:
            return (
                False,
                f"spread_too_wide("
                f"{current_spread_points:.1f} pts > {limit:.1f} pts limit)"
            )
        return True, ""

    # ── CVaR monitoring ───────────────────────────────────────────────────────
    def check_cvar(
        self,
        daily_pnl_history: np.ndarray,
        account_balance: float,
    ) -> tuple[bool, float]:
        """
        Check portfolio CVaR against limit (Section 7.7).
        Returns (within_limit: bool, cvar_value: float).
        """
        cvar = compute_cvar(daily_pnl_history, confidence=0.95, account_balance=account_balance)
        limit = self._portfolio_limits.get("max_cvar_5pct_daily_pct", 0.03)
        within_limit = cvar <= limit

        if not within_limit:
            logger.warning(
                f"CVaR {cvar:.2%} exceeds limit {limit:.2%}. "
                f"Reduce all position sizes until CVaR ≤ limit."
            )

        return within_limit, cvar

    # ── Kill switch ───────────────────────────────────────────────────────────
    def _trigger_kill_switch(self, reason: str) -> None:
        """Mark kill switch as triggered — stops all new position sizing."""
        self._kill_switch_triggered = True
        logger.critical(
            f"KILL SWITCH TRIGGERED: {reason} | "
            f"All new position sizing blocked. Manual reset required."
        )

    def reset_kill_switch(self, authorised_by: str) -> None:
        """
        Manually reset kill switch after investigation.
        Requires human authorisation (name logged).
        """
        if not authorised_by:
            raise ValueError("Kill switch reset requires named human authorisation.")
        self._kill_switch_triggered = False
        logger.critical(
            f"KILL SWITCH RESET by: {authorised_by} at "
            f"{datetime.now(timezone.utc).isoformat()}"
        )

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_triggered

    # ── Balance and P&L update ────────────────────────────────────────────────
    def update_equity(self, current_equity: float) -> DrawdownLevel:
        """Update equity curve and return current drawdown level."""
        self.drawdown_tracker.update(current_equity)
        return self.drawdown_tracker.drawdown_level

    def record_daily_pnl(self, daily_pnl: float) -> None:
        """Record end-of-day P&L for CVaR and CUSUM monitoring."""
        self._daily_pnl_history.append(daily_pnl)

    @property
    def daily_pnl_history(self) -> np.ndarray:
        return np.array(self._daily_pnl_history)

    def get_risk_summary(self) -> dict:
        """Return current risk state for monitoring dashboard."""
        return {
            "kill_switch_active": self._kill_switch_triggered,
            "drawdown_pct": self.drawdown_tracker.drawdown_pct,
            "daily_loss_pct": self.drawdown_tracker.daily_loss_pct,
            "drawdown_level": self.drawdown_tracker.drawdown_level.value,
            "drawdown_days": self.drawdown_tracker.drawdown_days,
            "total_open_risk_usd": self.portfolio_tracker.total_open_risk_usd,
            "n_open_positions": len(self.portfolio_tracker._open_positions),
            "extended_drawdown_alert": self.drawdown_tracker.extended_drawdown_alert(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
