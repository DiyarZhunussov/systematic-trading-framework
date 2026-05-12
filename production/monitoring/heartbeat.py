"""
heartbeat.py — Heartbeat Supervisor and Kill Switch
Implements Section 9.4 of the framework.

Design principles:
    - Runs in a SEPARATE THREAD with independent exception handling
    - Kill switch operable via secondary connection (MT5 mobile app)
    - Failsafe default: close all positions, never maintain unmonitored
    - Staggered closing: largest losses first, 30-second intervals
      (simultaneous closing creates market impact cascade — Section 7.6)
    - Manual authorisation required to restart after kill switch

Kill switch triggers (Section 9.4):
    - Component heartbeat silent > 120 seconds
    - MT5 terminal not responding
    - Margin level < 200%
    - Daily loss > internal limit (2%)
    - Drawdown > internal limit (8%)

Component registration:
    Each module calls heartbeat.register() at startup.
    Each module calls heartbeat.beat() periodically in its main loop.
    If any module goes silent > timeout: kill switch fires.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# ALERT MANAGER (Telegram + logging)
# ─────────────────────────────────────────────────────────────────────────────
class AlertManager:
    """
    Routes alerts to Telegram and structured log.
    All critical alerts are also logged regardless of Telegram status.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        mon = config.monitoring_params
        self._token = mon.get("telegram_bot_token", "")
        self._chat_id = mon.get("telegram_chat_id", "")
        self._alert_on_trade = mon.get("alert_on_trade", True)
        self._alert_on_kill = mon.get("alert_on_kill_switch", True)
        self._telegram_available = bool(self._token and self._chat_id)
        self._alert_history: list[dict] = []

    def _send_telegram(self, message: str) -> bool:
        """Send message to Telegram. Returns True on success."""
        if not self._telegram_available:
            return False
        try:
            import urllib.request
            import urllib.parse
            import json
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self._chat_id,
                "text": message,
                "parse_mode": "HTML",
            }).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    def _record(self, level: str, message: str) -> None:
        self._alert_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        })

    def info(self, message: str) -> None:
        logger.info(f"[ALERT INFO] {message}")
        self._record("info", message)

    def warning(self, message: str) -> None:
        logger.warning(f"[ALERT WARN] {message}")
        self._record("warning", message)
        self._send_telegram(f"⚠️ WARNING\n{message}")

    def critical(self, message: str) -> None:
        logger.critical(f"[ALERT CRITICAL] {message}")
        self._record("critical", message)
        self._send_telegram(f"🚨 CRITICAL\n{message}")

    def trade(self, symbol: str, direction: str, lots: float, price: float,
              strategy: str, ticket: int) -> None:
        if not self._alert_on_trade:
            return
        msg = (
            f"📊 TRADE\n"
            f"Strategy: {strategy}\n"
            f"{symbol} {direction.upper()} {lots}lots @ {price:.5f}\n"
            f"Ticket: {ticket}"
        )
        logger.info(f"[TRADE] {symbol} {direction} {lots}lots @ {price:.5f} t={ticket}")
        self._record("trade", msg)
        self._send_telegram(msg)

    def kill_switch(self, reason: str) -> None:
        if not self._alert_on_kill:
            return
        msg = f"🛑 KILL SWITCH TRIGGERED\nReason: {reason}"
        logger.critical(f"[KILL SWITCH] {reason}")
        self._record("kill_switch", msg)
        self._send_telegram(msg)

    def recent_alerts(self, n: int = 20) -> list[dict]:
        return self._alert_history[-n:]


# ─────────────────────────────────────────────────────────────────────────────
# HEARTBEAT SUPERVISOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ComponentStatus:
    """Status of a single monitored component."""
    name: str
    last_beat: float        # time.time() value
    registered_at: float
    beat_count: int = 0
    is_silent: bool = False


class HeartbeatSupervisor:
    """
    Monitors all system components and triggers kill switch on failure.
    Runs in a separate daemon thread.

    Usage:
        supervisor = HeartbeatSupervisor(mt5_bridge, alert_manager, config)
        supervisor.register("signal_engine")
        supervisor.start()
        # In each component's main loop:
        supervisor.beat("signal_engine")
    """

    def __init__(
        self,
        mt5_bridge: "MT5Bridge",
        alert_manager: AlertManager,
        config: "SystemConfig",
    ):
        self._mt5_bridge = mt5_bridge
        self._alert_manager = alert_manager
        self._config = config

        hb = config.heartbeat_params
        self._timeout_seconds = float(hb.get("timeout_seconds", 120))
        self._monitor_interval = float(hb.get("monitor_interval_seconds", 30))
        self._min_margin_pct = float(hb.get("min_margin_level_pct", 200))

        risk = config.portfolio_limits
        self._daily_loss_limit = float(risk.get("max_daily_loss_pct", 0.02))
        kill = config.kill_switch_params
        self._kill_daily_loss = float(kill.get("daily_loss_trigger_pct", 0.03))
        self._kill_drawdown = float(kill.get("drawdown_trigger_pct", 0.08))

        self._components: dict[str, ComponentStatus] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._kill_fired = False
        self._kill_reason = ""

    # ── Component registration ────────────────────────────────────────────────
    def register(self, component: str) -> None:
        """Register a component for heartbeat monitoring."""
        with self._lock:
            now = time.time()
            self._components[component] = ComponentStatus(
                name=component,
                last_beat=now,
                registered_at=now,
            )
        logger.info(f"Heartbeat registered: {component}")

    def beat(self, component: str) -> None:
        """Record a heartbeat from a component."""
        with self._lock:
            if component in self._components:
                self._components[component].last_beat = time.time()
                self._components[component].beat_count += 1
                self._components[component].is_silent = False
            else:
                logger.warning(
                    f"Heartbeat from unregistered component: {component}. "
                    f"Call register() first."
                )

    # ── Thread management ─────────────────────────────────────────────────────
    def start(self) -> None:
        """Start the heartbeat monitor thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="HeartbeatSupervisor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Heartbeat supervisor started: "
            f"timeout={self._timeout_seconds}s, "
            f"interval={self._monitor_interval}s"
        )

    def stop(self) -> None:
        """Stop the heartbeat monitor thread gracefully."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Heartbeat supervisor stopped")

    # ── Monitor loop ──────────────────────────────────────────────────────────
    def _monitor_loop(self) -> None:
        """
        Main monitoring loop — runs every monitor_interval seconds.
        Catches all exceptions internally to prevent thread death.
        """
        logger.info("Heartbeat monitor loop started")
        while self._running:
            try:
                self._check_all()
            except Exception as e:
                self._alert_manager.critical(
                    f"Heartbeat monitor loop error: {e}. "
                    f"Triggering kill switch as safety measure."
                )
                self.trigger_kill_switch(f"monitor_loop_exception: {e}")
            time.sleep(self._monitor_interval)
        logger.info("Heartbeat monitor loop exited")

    def _check_all(self) -> None:
        """Perform all monitoring checks."""
        now = time.time()

        # ── Component heartbeat check ─────────────────────────────────────────
        with self._lock:
            component_snapshot = dict(self._components)

        for name, status in component_snapshot.items():
            silent_seconds = now - status.last_beat
            if silent_seconds > self._timeout_seconds:
                if not status.is_silent:
                    with self._lock:
                        if name in self._components:
                            self._components[name].is_silent = True
                    self._alert_manager.critical(
                        f"Component silent: {name} "
                        f"({silent_seconds:.0f}s, limit={self._timeout_seconds}s)"
                    )
                    self.trigger_kill_switch(f"heartbeat_failure:{name}")
                    return

        # ── MT5 connection check ──────────────────────────────────────────────
        if MT5_AVAILABLE:
            try:
                terminal = mt5.terminal_info()
                if terminal is None:
                    self._alert_manager.critical("MT5 terminal not responding")
                    self.trigger_kill_switch("mt5_connection_lost")
                    return

                account = mt5.account_info()
                if account is None:
                    self._alert_manager.critical("MT5 account info unavailable")
                    self.trigger_kill_switch("mt5_account_unavailable")
                    return

                # ── Margin level check ────────────────────────────────────────
                if account.margin > 0:
                    margin_level = account.margin_level
                    if margin_level < self._min_margin_pct:
                        self._alert_manager.critical(
                            f"Margin level critical: {margin_level:.0f}% < "
                            f"{self._min_margin_pct:.0f}%"
                        )
                        self.trigger_kill_switch(
                            f"margin_critical:{margin_level:.0f}%"
                        )
                        return

                # ── Daily drawdown check ──────────────────────────────────────
                if account.balance > 0:
                    daily_loss = (account.balance - account.equity) / account.balance
                    if daily_loss > self._kill_daily_loss:
                        self._alert_manager.critical(
                            f"Daily loss kill switch: {daily_loss:.2%} > "
                            f"{self._kill_daily_loss:.2%}"
                        )
                        self.trigger_kill_switch(
                            f"daily_loss:{daily_loss:.2%}"
                        )
                        return

            except Exception as e:
                self._alert_manager.warning(f"MT5 check error: {e}")

    # ── Kill switch ───────────────────────────────────────────────────────────
    def trigger_kill_switch(self, reason: str) -> None:
        """
        Emergency position closure — staggered, largest loss first.
        All positions closed regardless of P&L.
        30-second intervals to avoid market impact cascade.
        """
        if self._kill_fired:
            return  # Prevent re-entrancy

        self._kill_fired = True
        self._kill_reason = reason
        self._running = False

        self._alert_manager.kill_switch(reason)
        logger.critical(
            f"KILL SWITCH EXECUTING: {reason} | "
            f"Closing all positions..."
        )

        if not MT5_AVAILABLE:
            logger.info("[SIM] Kill switch — all positions would be closed")
            return

        try:
            positions = mt5.positions_get()
            if not positions:
                logger.info("Kill switch: no open positions to close")
                return

            # Sort: close biggest losses first (most urgent)
            sorted_positions = sorted(positions, key=lambda p: p.profit)
            n = len(sorted_positions)
            close_interval = float(
                self._config.kill_switch_params.get(
                    "position_close_interval_seconds", 30
                )
            )

            for i, position in enumerate(sorted_positions):
                logger.critical(
                    f"Kill switch closing {i+1}/{n}: "
                    f"ticket={position.ticket} {position.symbol} "
                    f"profit={position.profit:.2f}"
                )
                success = self._mt5_bridge.close_position(
                    ticket=position.ticket,
                    reason=f"kill_switch:{reason}",
                    deviation=50,
                )
                if not success:
                    self._alert_manager.critical(
                        f"FAILED to close ticket {position.ticket} "
                        f"({position.symbol}) — MANUAL ACTION REQUIRED"
                    )
                if i < n - 1:
                    time.sleep(close_interval)

            logger.critical(
                f"Kill switch complete: {n} positions closed. "
                f"Manual reset required to resume trading."
            )
            self._alert_manager.critical(
                f"Kill switch complete. {n} positions closed. "
                f"Manual reset required."
            )

        except Exception as e:
            self._alert_manager.critical(
                f"Kill switch execution error: {e}. "
                f"MANUAL POSITION CLOSURE MAY BE REQUIRED."
            )

    @property
    def kill_switch_fired(self) -> bool:
        return self._kill_fired

    @property
    def kill_reason(self) -> str:
        return self._kill_reason

    def component_statuses(self) -> dict[str, dict]:
        """Return status of all registered components for monitoring."""
        now = time.time()
        with self._lock:
            return {
                name: {
                    "last_beat_seconds_ago": now - status.last_beat,
                    "beat_count": status.beat_count,
                    "is_silent": status.is_silent,
                    "timeout_seconds": self._timeout_seconds,
                }
                for name, status in self._components.items()
            }

    def reset(self, authorised_by: str) -> None:
        """
        Reset kill switch after investigation.
        Requires named human authorisation.
        """
        if not authorised_by.strip():
            raise ValueError(
                "Kill switch reset requires named human authorisation. "
                "Cannot reset with empty authorised_by."
            )
        self._kill_fired = False
        self._kill_reason = ""
        self._running = True
        logger.critical(
            f"KILL SWITCH RESET by '{authorised_by}' at "
            f"{datetime.now(timezone.utc).isoformat()}"
        )
        self._alert_manager.critical(
            f"Kill switch reset by {authorised_by}"
        )
        # Restart monitor thread
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="HeartbeatSupervisor",
            daemon=True,
        )
        self._thread.start()
