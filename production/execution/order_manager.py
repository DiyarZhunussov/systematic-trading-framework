"""
order_manager.py — Order Lifecycle Manager
Manages the full lifecycle of trades from signal to close.

Responsibilities:
    - Translate risk engine output into MT5 orders
    - Track open positions and their metadata
    - Monitor exit conditions per engine type
    - Enforce time stops and trailing stops
    - Log every trade event for IC tracking and audit
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    PENDING  = "pending"
    OPEN     = "open"
    CLOSING  = "closing"
    CLOSED   = "closed"
    FAILED   = "failed"


@dataclass
class ManagedPosition:
    """Full metadata for a tracked position."""
    ticket: int
    symbol: str
    strategy: str
    direction: str                  # "buy" or "sell"
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    dollar_risk: float
    signal_confidence: float
    regime_at_entry: str
    entry_z_score: Optional[float]  # Mean reversion only
    atr_at_entry: float
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None
    realised_pnl: Optional[float] = None
    bars_held: int = 0

    @property
    def is_long(self) -> bool:
        return self.direction == "buy"

    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN

    def hours_held(self) -> float:
        now = datetime.now(timezone.utc)
        opened = self.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return (now - opened).total_seconds() / 3600

    def unrealised_pnl(self, current_price: float, contract_size: float = 100_000) -> float:
        if self.is_long:
            return (current_price - self.entry_price) * self.volume * contract_size
        return (self.entry_price - current_price) * self.volume * contract_size


class OrderManager:
    """
    Manages trade lifecycle from risk-engine approval to close.

    Usage:
        om = OrderManager(mt5_bridge, risk_engine, alert_manager)
        ticket = om.open_position(size_result, signal, instrument_config)
        om.check_exits(bar_arrays)   # called each bar
    """

    def __init__(
        self,
        mt5_bridge: "MT5Bridge",
        risk_engine: "RiskEngine",
        portfolio_tracker: "PortfolioRiskTracker",
        alert_manager: "AlertManager",
    ):
        self._mt5 = mt5_bridge
        self._risk = risk_engine
        self._portfolio = portfolio_tracker
        self._alerts = alert_manager
        self._positions: dict[int, ManagedPosition] = {}  # ticket → position
        self._closed_positions: list[ManagedPosition] = []
        self._trade_log: list[dict] = []

    # ── Open position ─────────────────────────────────────────────────────────
    def open_position(
        self,
        size_result: "PositionSizeResult",
        instrument_config: dict,
        take_profit: float = 0.0,
        entry_z_score: Optional[float] = None,
        comment: str = "",
    ) -> Optional[int]:
        """
        Submit an order and register the position for tracking.
        Returns ticket number on success, None on failure.
        """
        req = size_result.request

        if not size_result.is_approved or size_result.position_size_lots <= 0:
            logger.warning(
                f"Cannot open position: size_result not approved "
                f"({size_result.rejection_reason})"
            )
            return None

        result = self._mt5.execute_order(
            symbol=req.instrument,
            order_type=req.direction,
            volume=size_result.position_size_lots,
            stop_loss=req.stop_loss_price,
            take_profit=take_profit,
            strategy=req.strategy,
            comment=comment or req.strategy,
        )

        if result.get("status") not in ("filled", "simulation"):
            logger.error(
                f"Order failed: {req.instrument} | "
                f"reason={result.get('reason', result.get('comment', 'unknown'))}"
            )
            return None

        ticket = result.get("ticket", -1)
        actual_price = result.get("price", req.entry_price)
        atr = size_result.stop_distance  # Use stop distance as ATR proxy

        position = ManagedPosition(
            ticket=ticket,
            symbol=req.instrument,
            strategy=req.strategy,
            direction=req.direction,
            volume=size_result.position_size_lots,
            entry_price=actual_price,
            stop_loss=req.stop_loss_price,
            take_profit=take_profit,
            dollar_risk=size_result.dollar_risk,
            signal_confidence=req.signal_confidence,
            regime_at_entry=req.current_regime,
            entry_z_score=entry_z_score,
            atr_at_entry=atr,
        )
        self._positions[ticket] = position

        # Register with portfolio tracker
        self._portfolio.add_position(
            ticket=ticket,
            instrument=req.instrument,
            strategy=req.strategy,
            direction=req.direction,
            size_lots=size_result.position_size_lots,
            entry_price=actual_price,
            stop_price=req.stop_loss_price,
            dollar_risk=size_result.dollar_risk,
        )

        self._log_trade_event("open", position, result)
        self._alerts.trade(
            symbol=req.instrument,
            direction=req.direction,
            lots=size_result.position_size_lots,
            price=actual_price,
            strategy=req.strategy,
            ticket=ticket,
        )

        logger.info(
            f"Position opened: {req.instrument} {req.direction} "
            f"{size_result.position_size_lots}lots @ {actual_price:.5f} | "
            f"ticket={ticket} strategy={req.strategy}"
        )
        return ticket

    # ── Close position ────────────────────────────────────────────────────────
    def close_position(
        self,
        ticket: int,
        reason: str,
        current_price: Optional[float] = None,
    ) -> bool:
        """Close a position and update tracking."""
        position = self._positions.get(ticket)
        if position is None:
            logger.warning(f"close_position: ticket {ticket} not found in tracker")
            return False

        if position.status != PositionStatus.OPEN:
            logger.warning(
                f"close_position: ticket {ticket} status is {position.status.value}"
            )
            return False

        position.status = PositionStatus.CLOSING
        success = self._mt5.close_position(ticket, reason=reason)

        if success:
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now(timezone.utc)
            position.close_reason = reason
            if current_price is not None:
                position.close_price = current_price
                contract_size = 100_000  # Default; ideally from instrument config
                position.realised_pnl = position.unrealised_pnl(
                    current_price, contract_size
                )

            self._portfolio.remove_position(ticket)
            self._closed_positions.append(position)
            del self._positions[ticket]

            self._log_trade_event("close", position)
            logger.info(
                f"Position closed: {position.symbol} ticket={ticket} "
                f"reason={reason} pnl={position.realised_pnl}"
            )
        else:
            position.status = PositionStatus.OPEN  # Revert — close failed
            self._alerts.critical(
                f"CLOSE FAILED: ticket={ticket} {position.symbol} reason={reason}"
            )

        return success

    # ── Exit monitoring ───────────────────────────────────────────────────────
    def check_exits(
        self,
        bar_arrays: dict[str, "BarArray"],
        mean_rev_engine: Optional["MeanReversionEngine"] = None,
        breakout_engine: Optional["VolatilityBreakoutEngine"] = None,
    ) -> list[int]:
        """
        Check exit conditions for all open positions.
        Called each bar by the main orchestrator.
        Returns list of ticket numbers that were closed.
        """
        closed_tickets = []

        for ticket, position in list(self._positions.items()):
            if position.status != PositionStatus.OPEN:
                continue

            position.bars_held += 1
            bar_array = bar_arrays.get(position.symbol)

            if bar_array is None or bar_array.n < 2:
                continue

            current_price = float(bar_array.close[-1])
            should_close = False
            close_reason = ""

            # ── Mean reversion exit logic ─────────────────────────────────────
            if position.strategy == "mean_reversion" and mean_rev_engine:
                from production.engines.signal_engine_mean_reversion import SignalDirection
                direction = (
                    SignalDirection.LONG if position.is_long
                    else SignalDirection.SHORT
                )
                should_close, close_reason = mean_rev_engine.should_exit(
                    instrument=position.symbol,
                    entry_z_adj=position.entry_z_score or 0.0,
                    current_prices=bar_array.close,
                    current_timestamps=bar_array.timestamps,
                    entry_timestamp=position.opened_at,
                    direction=direction,
                )

            # ── Breakout exit logic ───────────────────────────────────────────
            elif position.strategy == "volatility_breakout" and breakout_engine:
                from production.engines.signal_engine_volatility_breakout import SignalDirection
                direction = (
                    SignalDirection.LONG if position.is_long
                    else SignalDirection.SHORT
                )
                should_close, close_reason = breakout_engine.should_exit(
                    instrument=position.symbol,
                    direction=direction,
                    entry_price=position.entry_price,
                    current_price=current_price,
                    atr_at_entry=position.atr_at_entry,
                    bars_held=position.bars_held,
                )

            # ── Generic stop/target check ─────────────────────────────────────
            if not should_close:
                if position.is_long:
                    if current_price <= position.stop_loss:
                        should_close = True
                        close_reason = f"stop_loss({current_price:.5f})"
                    elif position.take_profit > 0 and current_price >= position.take_profit:
                        should_close = True
                        close_reason = f"take_profit({current_price:.5f})"
                else:
                    if current_price >= position.stop_loss:
                        should_close = True
                        close_reason = f"stop_loss({current_price:.5f})"
                    elif position.take_profit > 0 and current_price <= position.take_profit:
                        should_close = True
                        close_reason = f"take_profit({current_price:.5f})"

            if should_close:
                success = self.close_position(ticket, close_reason, current_price)
                if success:
                    closed_tickets.append(ticket)

        return closed_tickets

    # ── Reconciliation ────────────────────────────────────────────────────────
    def reconcile_with_mt5(self) -> dict:
        """
        Reconcile internal position tracker with MT5's actual positions.
        Detects and reports discrepancies (positions closed externally,
        unexpected positions from other sources, etc.)
        Called at startup and periodically.
        """
        mt5_positions = {p["ticket"]: p for p in self._mt5.get_open_positions()}
        internal_tickets = set(self._positions.keys())
        mt5_tickets = set(mt5_positions.keys())

        # In MT5 but not in tracker — externally opened or tracker missed close
        unexpected = mt5_tickets - internal_tickets
        # In tracker but not in MT5 — externally closed (stop hit, manual close)
        missing = internal_tickets - mt5_tickets

        for ticket in missing:
            position = self._positions[ticket]
            logger.warning(
                f"RECONCILIATION: ticket {ticket} ({position.symbol}) "
                f"not in MT5 — marking closed externally"
            )
            position.status = PositionStatus.CLOSED
            position.closed_at = datetime.now(timezone.utc)
            position.close_reason = "closed_externally"
            self._portfolio.remove_position(ticket)
            self._closed_positions.append(position)
            del self._positions[ticket]

        if unexpected:
            logger.warning(
                f"RECONCILIATION: {len(unexpected)} unexpected MT5 positions "
                f"(not from this system): {unexpected}"
            )

        return {
            "unexpected_in_mt5": list(unexpected),
            "missing_from_mt5": list(missing),
            "internal_count": len(self._positions),
            "mt5_count": len(mt5_positions),
        }

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log_trade_event(
        self,
        event_type: str,
        position: ManagedPosition,
        fill_result: Optional[dict] = None,
    ) -> None:
        entry = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket": position.ticket,
            "symbol": position.symbol,
            "strategy": position.strategy,
            "direction": position.direction,
            "volume": position.volume,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "dollar_risk": position.dollar_risk,
            "signal_confidence": position.signal_confidence,
            "regime": position.regime_at_entry,
        }
        if event_type == "close":
            entry.update({
                "close_price": position.close_price,
                "close_reason": position.close_reason,
                "realised_pnl": position.realised_pnl,
                "bars_held": position.bars_held,
                "hours_held": position.hours_held(),
            })
        if fill_result:
            entry.update({
                "slippage_pts": fill_result.get("slippage_pts"),
                "latency_ms": fill_result.get("latency_ms"),
            })
        self._trade_log.append(entry)

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def open_positions(self) -> dict[int, ManagedPosition]:
        return {t: p for t, p in self._positions.items()
                if p.status == PositionStatus.OPEN}

    @property
    def trade_log(self) -> list[dict]:
        return list(self._trade_log)

    @property
    def closed_positions(self) -> list[ManagedPosition]:
        return list(self._closed_positions)
