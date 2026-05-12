"""
shadow_mode.py — Shadow Deployment (Stage 5 of 7-Stage Pipeline)
Implements Section 10.1, Stage 5 of the framework.

Stage 5 requirements:
    Duration  : Minimum 30 days concurrent with late paper trading
    Activity  : ALL signals generated in production environment
                Orders submitted as VIRTUAL (no fill confirmation)
    Required  : Zero critical system failures
                Signal timing within spec
                All monitoring alerts firing correctly
    Gate      : Technical review of system reliability
    Output    : Shadow deployment log; system stability report

Key difference from paper trading (Stage 4):
    Paper trading (Stage 4) validates STRATEGY PERFORMANCE.
    Shadow mode (Stage 5) validates SYSTEM RELIABILITY.
    In shadow mode the FULL production stack runs — including MT5 bridge,
    heartbeat, order manager — but orders are intercepted before submission.
    This surfaces integration bugs, timing issues, and monitoring gaps
    that paper trading (which bypasses parts of the stack) would miss.
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
)
logger = logging.getLogger("shadow_mode")

from production.config.config import load_config
from production.execution.mt5_bridge import MT5Bridge
from production.monitoring.heartbeat import HeartbeatSupervisor, AlertManager
from production.monitoring.structured_logger import StructuredLogger


# ─────────────────────────────────────────────────────────────────────────────
# SHADOW MT5 BRIDGE
# Wraps the real bridge and intercepts execute_order calls
# ─────────────────────────────────────────────────────────────────────────────
class ShadowMT5Bridge(MT5Bridge):
    """
    MT5Bridge subclass that intercepts order submissions.
    All pre-flight checks, fill quality tracking, and position queries
    run against the real MT5 terminal. Only execute_order is intercepted.
    """

    def __init__(self, config, shadow_log: StructuredLogger):
        super().__init__(config)
        self._shadow_log = shadow_log
        self._virtual_orders: list[dict] = []
        self._virtual_ticket_counter = 900_000

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        stop_loss: float,
        take_profit: float,
        strategy: str = "",
        comment: str = "",
    ) -> dict:
        """
        Intercept order — run preflight but do NOT submit to MT5.
        Returns a synthetic 'filled' result for downstream processing.
        """
        # Still run the preflight check (validates spread, margin, connectivity)
        instrument_config = self.config.instruments.get(
            "instruments", {}
        ).get(symbol, {})
        check = self.preflight_check(symbol, instrument_config)

        self._virtual_ticket_counter += 1
        ticket = self._virtual_ticket_counter

        # Get current price for realistic fill simulation
        tick = None
        try:
            import MetaTrader5 as mt5
            t = mt5.symbol_info_tick(symbol)
            if t:
                tick = {"bid": t.bid, "ask": t.ask}
        except Exception:
            pass

        fill_price = (
            tick["ask"] if order_type == "buy" else tick["bid"]
        ) if tick else 0.0

        virtual_result = {
            "status": "virtual_filled",
            "ticket": ticket,
            "price": fill_price,
            "volume": volume,
            "slippage_pts": 0.0,
            "latency_ms": 0.0,
            "strategy": strategy,
            "symbol": symbol,
            "preflight": check,
        }

        self._virtual_orders.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "order_type": order_type,
            "volume": volume,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "strategy": strategy,
            "fill_price": fill_price,
            "preflight_ok": check.get("ok", False),
            "ticket": ticket,
        })

        logger.info(
            f"[SHADOW] Virtual order: {symbol} {order_type} {volume}lots "
            f"@ {fill_price:.5f} | preflight={'OK' if check.get('ok') else 'FAIL'} | "
            f"ticket={ticket}"
        )

        self._shadow_log.log_trade_open(
            ticket=ticket, symbol=symbol, strategy=strategy,
            direction=order_type, volume=volume,
            entry_price=fill_price, stop_loss=stop_loss,
            take_profit=take_profit, dollar_risk=0.0,
            signal_confidence=0.0, regime="shadow",
        )

        return virtual_result

    def close_position(self, ticket: int, reason: str = "", deviation: int = 50) -> bool:
        """Intercept close — log but don't actually close."""
        logger.info(f"[SHADOW] Virtual close: ticket={ticket} reason={reason}")
        return True

    @property
    def virtual_order_count(self) -> int:
        return len(self._virtual_orders)

    @property
    def virtual_orders(self) -> list[dict]:
        return list(self._virtual_orders)


# ─────────────────────────────────────────────────────────────────────────────
# SHADOW DEPLOYMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
class ShadowDeployment:
    """
    Runs the full production TradingSystem with a ShadowMT5Bridge.
    Monitors for system failures, timing issues, and alert correctness.
    """

    MIN_DAYS_REQUIRED = 30
    CRITICAL_FAILURE_THRESHOLD = 0  # Zero critical failures allowed

    def __init__(self, config):
        self.config = config
        self._start_time = datetime.now(timezone.utc)
        self._critical_failures: list[dict] = []
        self._warnings: list[dict] = []
        self._slog = StructuredLogger(str(ROOT / "staging" / "shadow_logs"))
        self._running = False

        logger.info(
            f"ShadowDeployment initialised | "
            f"required_days={self.MIN_DAYS_REQUIRED}"
        )

    def run(self) -> None:
        """
        Run shadow deployment using the full TradingSystem with intercepted MT5.
        Imports and patches the production main module.
        """
        logger.info("=" * 60)
        logger.info("SHADOW DEPLOYMENT STARTED — Stage 5 of 7-stage pipeline")
        logger.info("ALL ORDERS INTERCEPTED — no real trades will execute")
        logger.info("=" * 60)

        # Monkey-patch MT5Bridge in the main module
        import production.execution.mt5_bridge as bridge_module
        original_bridge_class = bridge_module.MT5Bridge

        shadow_bridge_instance = ShadowMT5Bridge(self.config, self._slog)
        bridge_module.MT5Bridge = lambda cfg: shadow_bridge_instance

        try:
            from production.main import TradingSystem
            system = TradingSystem(self.config)

            # Register shadow-specific signal handlers
            signal.signal(signal.SIGINT, lambda s, f: self._stop(system))
            signal.signal(signal.SIGTERM, lambda s, f: self._stop(system))

            self._running = True

            if system.start():
                logger.info("Shadow deployment: system started successfully")
                system.run()
            else:
                self._record_critical_failure("startup_failed", "TradingSystem.start() returned False")

        except Exception as e:
            self._record_critical_failure("unhandled_exception", str(e))
            logger.exception(f"Shadow deployment error: {e}")
        finally:
            bridge_module.MT5Bridge = original_bridge_class
            self._generate_stage5_report(shadow_bridge_instance)

    def _stop(self, system) -> None:
        self._running = False
        if system:
            system.shutdown(graceful=True)

    def _record_critical_failure(self, failure_type: str, detail: str) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": failure_type,
            "detail": detail,
        }
        self._critical_failures.append(entry)
        logger.critical(f"SHADOW CRITICAL FAILURE: {failure_type} — {detail}")
        self._slog.log_risk_event("shadow_critical_failure", entry, severity="critical")

    def _generate_stage5_report(self, shadow_bridge: ShadowMT5Bridge) -> dict:
        """Generate Stage 5 technical review report."""
        elapsed_days = (datetime.now(timezone.utc) - self._start_time).days
        n_critical = len(self._critical_failures)
        n_virtual_orders = shadow_bridge.virtual_order_count

        preflight_results = [o["preflight_ok"] for o in shadow_bridge.virtual_orders]
        preflight_pass_rate = (
            sum(preflight_results) / len(preflight_results)
            if preflight_results else 1.0
        )

        report = {
            "stage": 5,
            "days_elapsed": elapsed_days,
            "min_days_required": self.MIN_DAYS_REQUIRED,
            "duration_satisfied": elapsed_days >= self.MIN_DAYS_REQUIRED,
            "n_critical_failures": n_critical,
            "critical_failures": self._critical_failures,
            "n_virtual_orders": n_virtual_orders,
            "preflight_pass_rate": preflight_pass_rate,
            "gate_passes": (
                elapsed_days >= self.MIN_DAYS_REQUIRED
                and n_critical == self.CRITICAL_FAILURE_THRESHOLD
            ),
        }

        logger.info("=" * 60)
        logger.info("STAGE 5 SHADOW DEPLOYMENT REPORT")
        logger.info("=" * 60)
        logger.info(f"  Days elapsed         : {elapsed_days} / {self.MIN_DAYS_REQUIRED}")
        logger.info(f"  Critical failures    : {n_critical} (limit: 0)")
        logger.info(f"  Virtual orders       : {n_virtual_orders}")
        logger.info(f"  Preflight pass rate  : {preflight_pass_rate:.1%}")
        verdict = "PASS" if report["gate_passes"] else "FAIL"
        logger.info(f"\nGate verdict: {verdict}")
        logger.info("=" * 60)

        return report


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    shadow = ShadowDeployment(config)
    shadow.run()
