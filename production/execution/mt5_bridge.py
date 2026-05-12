"""
mt5_bridge.py — MetaTrader 5 Bridge
Production-grade MT5 integration with fill quality monitoring.
Implements Section 9.3 of the framework.

Architecture decision:
    Uses MetaTrader5 Python library directly (no ZeroMQ/REST overhead).
    All strategies operate on 4H/daily bars — MT5 async benefits unnecessary.
    Fill quality tracking monitors latency and slippage deterioration.

VPS latency guidance (Section 2.2 — corrected from adversarial review):
    Target: sub-20ms round-trip to broker's matching engine.
    IC Markets / Pepperstone US  → Equinix NY4 (Secaucus, NJ)
    IC Markets AU / Pepperstone AU → Equinix LD4 (Slough, UK)
    Asian session brokers         → Equinix TY3 (Tokyo)
    "Same city" VPS is insufficient if routing via public internet.

Fill quality (Section 2.2):
    Each 50ms of additional latency ≈ 0.1–0.5 pip excess slippage on EUR/USD.
    At 5–8 pip gross edge: 1–10% degradation per trade.
    This degrades the strategy but does not destroy it unless edge is already marginal.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available — MT5Bridge running in simulation mode")


# ─────────────────────────────────────────────────────────────────────────────
# FILL RECORD
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FillRecord:
    """Detailed record of a single order fill for quality monitoring."""
    ticket: int
    symbol: str
    order_type: str             # "buy" or "sell"
    strategy: str
    expected_price: float
    actual_price: float
    signal_time: datetime
    execution_time: datetime
    slippage_pts: float         # Points of slippage (signed — positive = adverse)
    latency_ms: float
    spread_pts: float
    volume: float
    comment: str = ""

    @property
    def adverse_slippage(self) -> bool:
        """True if slippage moved against the trade."""
        if self.order_type == "buy":
            return self.actual_price > self.expected_price
        return self.actual_price < self.expected_price


# ─────────────────────────────────────────────────────────────────────────────
# FILL TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class FillTracker:
    """
    Monitors fill quality over time and alerts on deterioration.
    Implements Condition D of the alert schedule (Section 6.6):
        Slippage deterioration > 20% vs 90-day baseline → alert.
    """

    def __init__(self, alert_threshold_pct: float = 0.20):
        """
        Parameters
        ----------
        alert_threshold_pct : Alert when recent slippage exceeds
                              baseline by this fraction (default 20%)
        """
        self._records: list[FillRecord] = []
        self._alert_threshold = alert_threshold_pct
        self._baseline_established = False
        self._last_alert: Optional[datetime] = None

    def record(self, fill: FillRecord) -> bool:
        """
        Record a fill and check for deterioration.
        Returns True if a deterioration alert was raised.
        """
        self._records.append(fill)
        alerted = self._check_deterioration()
        return alerted

    def _check_deterioration(self) -> bool:
        """
        Compare recent 20-trade slippage against first-100-trade baseline.
        Returns True if alert condition is met.
        """
        if len(self._records) < 100:
            return False  # Insufficient baseline

        baseline_slips = [abs(r.slippage_pts) for r in self._records[:100]]
        recent_slips = [abs(r.slippage_pts) for r in self._records[-20:]]

        baseline = np.mean(baseline_slips)
        recent = np.mean(recent_slips)

        if baseline < 1e-10:
            return False

        ratio = recent / baseline
        if ratio > (1 + self._alert_threshold):
            # Rate-limit alerts — don't spam every bar
            now = datetime.now(timezone.utc)
            if (
                self._last_alert is None
                or (now - self._last_alert).total_seconds() > 3600
            ):
                self._last_alert = now
                logger.warning(
                    f"FILL QUALITY ALERT: Slippage deterioration detected. "
                    f"Recent={recent:.2f}pts, Baseline={baseline:.2f}pts, "
                    f"Ratio={ratio:.2f} (threshold={1+self._alert_threshold:.2f}). "
                    f"Condition D active."
                )
                return True
        return False

    def recent_avg_slippage(self, n: int = 50) -> Optional[float]:
        if not self._records:
            return None
        recent = [abs(r.slippage_pts) for r in self._records[-n:]]
        return float(np.mean(recent))

    def recent_avg_latency_ms(self, n: int = 50) -> Optional[float]:
        if not self._records:
            return None
        recent = [r.latency_ms for r in self._records[-n:]]
        return float(np.mean(recent))

    def slippage_ratio(self) -> float:
        """recent_avg / baseline_avg — 1.0 means no deterioration."""
        if len(self._records) < 100:
            return 1.0
        baseline = np.mean([abs(r.slippage_pts) for r in self._records[:100]])
        recent = np.mean([abs(r.slippage_pts) for r in self._records[-50:]])
        if baseline < 1e-10:
            return 1.0
        return float(recent / baseline)

    def get_stats(self) -> dict:
        """Return fill quality statistics for monitoring dashboard."""
        if not self._records:
            return {"n_fills": 0}

        slips = [abs(r.slippage_pts) for r in self._records]
        latencies = [r.latency_ms for r in self._records]
        adverse = sum(1 for r in self._records if r.adverse_slippage)

        return {
            "n_fills": len(self._records),
            "avg_slippage_pts": float(np.mean(slips)),
            "max_slippage_pts": float(np.max(slips)),
            "p95_slippage_pts": float(np.percentile(slips, 95)),
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "adverse_fill_rate": adverse / len(self._records),
            "slippage_ratio": self.slippage_ratio(),
            "deterioration_alert": self.slippage_ratio() > (1 + self._alert_threshold),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MT5 BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
class MT5Bridge:
    """
    Production MT5 integration with pre-flight checks and fill tracking.

    Pre-flight checks before every order:
        1. Symbol available
        2. Spread within limit
        3. Margin level adequate
        4. Account connected

    Fill tracking:
        Every fill recorded with latency and slippage.
        Deterioration alerts trigger Condition D monitoring.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        self._mt5_cfg = config.mt5
        self.fill_tracker = FillTracker(
            alert_threshold_pct=float(
                self._mt5_cfg.get("slippage_alert_threshold", 0.20)
            )
        )
        self._connected = False
        self._magic = int(self._mt5_cfg.get("magic_number", 20250101))

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> bool:
        """
        Establish MT5 connection with retry logic.
        Returns True on success.
        """
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available — bridge in simulation mode")
            self._connected = False
            return False

        max_retries = int(self._mt5_cfg.get("max_retries", 3))
        retry_delay = float(self._mt5_cfg.get("retry_delay_seconds", 5))

        for attempt in range(1, max_retries + 1):
            try:
                success = mt5.initialize(
                    path=self._mt5_cfg.get("path", ""),
                    login=int(self._mt5_cfg.get("login", 0)),
                    password=str(self._mt5_cfg.get("password", "")),
                    server=str(self._mt5_cfg.get("server", "")),
                )
                if success:
                    self._connected = True
                    info = mt5.terminal_info()
                    account = mt5.account_info()
                    logger.info(
                        f"MT5 connected: build={info.build if info else 'unknown'} | "
                        f"ping={info.ping_last if info else 'unknown'}ms | "
                        f"account={account.login if account else 'unknown'} | "
                        f"balance={account.balance if account else 'unknown'}"
                    )
                    return True

                err = mt5.last_error() if MT5_AVAILABLE else ("", "")
                logger.warning(
                    f"MT5 connection attempt {attempt}/{max_retries} failed: {err}"
                )
            except Exception as e:
                logger.warning(f"MT5 connect exception attempt {attempt}: {e}")

            if attempt < max_retries:
                time.sleep(retry_delay)

        self._connected = False
        logger.error("MT5 connection failed after all retries")
        return False

    def disconnect(self) -> None:
        if MT5_AVAILABLE:
            mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        info = mt5.terminal_info()
        return info is not None and info.connected

    # ── Pre-flight check ──────────────────────────────────────────────────────
    def preflight_check(
        self,
        symbol: str,
        instrument_config: dict,
    ) -> dict:
        """
        Validate market conditions before order submission.
        Returns {'ok': bool, 'reason': str, ...}
        """
        if not MT5_AVAILABLE:
            return {"ok": False, "reason": "mt5_not_available"}

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)

        if tick is None or info is None:
            return {"ok": False, "reason": "symbol_unavailable"}

        # Spread check
        spread_pts = (tick.ask - tick.bid) / (info.point + 1e-10)
        max_spread = instrument_config.get("max_spread_points", 999)
        if spread_pts > max_spread:
            return {
                "ok": False,
                "reason": f"spread_too_wide({spread_pts:.1f} > {max_spread}pts)",
                "spread_pts": spread_pts,
                "max_spread": max_spread,
            }

        # Margin level check
        account = mt5.account_info()
        if account is None:
            return {"ok": False, "reason": "account_info_unavailable"}

        min_margin = float(
            self.config.heartbeat_params.get("min_margin_level_pct", 200)
        )
        if account.margin > 0 and account.margin_level < min_margin:
            return {
                "ok": False,
                "reason": f"margin_level_critical({account.margin_level:.0f}% < {min_margin:.0f}%)",
            }

        return {
            "ok": True,
            "spread_pts": spread_pts,
            "bid": tick.bid,
            "ask": tick.ask,
            "account_balance": account.balance,
            "account_equity": account.equity,
        }

    # ── Order execution ───────────────────────────────────────────────────────
    def execute_order(
        self,
        symbol: str,
        order_type: str,         # "buy" or "sell"
        volume: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
        comment: str = "",
    ) -> dict:
        """
        Execute a market order with pre-flight checks and fill tracking.

        Returns dict with status, ticket, price, slippage, latency.
        """
        if not MT5_AVAILABLE:
            return {"status": "simulation", "reason": "mt5_not_available"}

        instrument_config = self.config.instruments.get(
            "instruments", {}
        ).get(symbol, {})

        # Pre-flight
        check = self.preflight_check(symbol, instrument_config)
        if not check["ok"]:
            logger.warning(f"Pre-flight rejected {symbol} {order_type}: {check['reason']}")
            return {"status": "rejected", **check}

        tick = mt5.symbol_info_tick(symbol)
        expected_price = tick.ask if order_type == "buy" else tick.bid
        signal_time = datetime.now(timezone.utc)

        order_type_mt5 = (
            mt5.ORDER_TYPE_BUY if order_type == "buy"
            else mt5.ORDER_TYPE_SELL
        )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_mt5,
            "price": expected_price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": 20,
            "magic": self._magic,
            "comment": comment or f"{strategy}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        execution_time = datetime.now(timezone.utc)

        if result is None:
            logger.error(f"order_send returned None for {symbol}: {mt5.last_error()}")
            return {
                "status": "failed",
                "reason": "null_result",
                "mt5_error": str(mt5.last_error()),
            }

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            info = mt5.symbol_info(symbol)
            point = info.point if info else 0.0001
            slippage_pts = (result.price - expected_price) / (point + 1e-10)
            latency_ms = (
                execution_time - signal_time
            ).total_seconds() * 1000

            fill = FillRecord(
                ticket=result.order,
                symbol=symbol,
                order_type=order_type,
                strategy=strategy,
                expected_price=expected_price,
                actual_price=result.price,
                signal_time=signal_time,
                execution_time=execution_time,
                slippage_pts=float(slippage_pts),
                latency_ms=float(latency_ms),
                spread_pts=float(check.get("spread_pts", 0)),
                volume=float(volume),
                comment=comment,
            )
            self.fill_tracker.record(fill)

            logger.info(
                f"FILLED: {symbol} {order_type} {volume}lots "
                f"@ {result.price:.5f} | "
                f"slip={slippage_pts:.1f}pts | lat={latency_ms:.0f}ms | "
                f"ticket={result.order}"
            )

            return {
                "status": "filled",
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume,
                "slippage_pts": float(slippage_pts),
                "latency_ms": float(latency_ms),
                "strategy": strategy,
                "symbol": symbol,
            }

        # Failed
        logger.error(
            f"Order failed: {symbol} {order_type} | "
            f"retcode={result.retcode} | {result.comment}"
        )
        return {
            "status": "failed",
            "retcode": result.retcode,
            "comment": result.comment,
            "symbol": symbol,
        }

    # ── Modify position stop ──────────────────────────────────────────────────
    def modify_position(
        self,
        ticket: int,
        new_sl: float,
        new_tp: float = 0.0,
    ) -> bool:
        """Modify stop loss (and optionally take profit) of open position."""
        if not MT5_AVAILABLE:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(new_sl),
            "tp": float(new_tp),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Modified ticket {ticket}: SL={new_sl:.5f}")
            return True

        logger.error(
            f"Modify failed for ticket {ticket}: "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    # ── Close position ────────────────────────────────────────────────────────
    def close_position(
        self,
        ticket: int,
        reason: str = "",
        deviation: int = 50,
    ) -> bool:
        """
        Close a specific position by ticket number.
        Used by kill switch — deviation=50 for urgent closes.
        """
        if not MT5_AVAILABLE:
            logger.info(f"[SIM] Close position ticket={ticket} reason={reason}")
            return True

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position ticket {ticket} not found")
            return False

        pos = positions[0]
        close_type = (
            mt5.ORDER_TYPE_SELL if pos.type == 0
            else mt5.ORDER_TYPE_BUY
        )
        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            logger.error(f"No tick for {pos.symbol} when closing {ticket}")
            return False

        close_price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": deviation,
            "magic": self._magic,
            "comment": f"close:{reason}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"CLOSED ticket={ticket} {pos.symbol} "
                f"@ {result.price:.5f} | reason={reason}"
            )
            return True

        logger.error(
            f"Close failed for ticket {ticket}: "
            f"retcode={result.retcode if result else 'None'}"
        )
        return False

    # ── Position queries ──────────────────────────────────────────────────────
    def get_open_positions(self) -> list[dict]:
        """Return all open positions as list of dicts."""
        if not MT5_AVAILABLE:
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "buy" if p.type == 0 else "sell",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "magic": p.magic,
                "comment": p.comment,
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc),
            }
            for p in positions
            if p.magic == self._magic  # Only our positions
        ]

    def get_account_info(self) -> Optional[dict]:
        """Return current account balance, equity, margin."""
        if not MT5_AVAILABLE:
            return None
        acc = mt5.account_info()
        if acc is None:
            return None
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "margin_level": acc.margin_level,
            "profit": acc.profit,
            "currency": acc.currency,
        }

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Return symbol specification."""
        if not MT5_AVAILABLE:
            return None
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "symbol": info.name,
            "point": info.point,
            "digits": info.digits,
            "contract_size": info.trade_contract_size,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
            "spread": info.spread,
            "trade_allowed": info.trade_mode != 0,
        }
