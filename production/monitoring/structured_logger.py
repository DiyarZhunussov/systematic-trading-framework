"""
structured_logger.py — Structured JSON Logger
Writes all system events to rotating JSON log files.

Log categories (Section 9.2 folder structure):
    trades/   — every trade open/close with full metadata
    signals/  — signal generation with IC tracking
    risk/     — risk events, drawdown, kill switch
    system/   — process health, latency, connection

Design:
    - Each log entry is a single-line JSON (JSONL format)
    - Rotating by date: one file per day per category
    - Never blocks the main trading loop (buffered writes)
    - Flush on every write in production (no buffering loss on crash)
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    Thread-safe structured JSON logger.
    Writes JSONL (one JSON object per line) to category-based log files.
    """

    CATEGORIES = ("trades", "signals", "risk", "system")

    def __init__(self, log_base_dir: str = "production/logs"):
        self._base = Path(log_base_dir)
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}

        # Ensure log directories exist
        for cat in self.CATEGORIES:
            (self._base / cat).mkdir(parents=True, exist_ok=True)

    def _get_handle(self, category: str) -> Any:
        """Return (or open) today's log file handle for a category."""
        today = datetime.now(timezone.utc).date().isoformat()
        key = f"{category}::{today}"

        if key not in self._handles:
            # Close yesterday's handle if open
            for k in list(self._handles):
                if k.startswith(f"{category}::") and k != key:
                    try:
                        self._handles[k].close()
                    except Exception:
                        pass
                    del self._handles[k]

            path = self._base / category / f"{today}.jsonl"
            self._handles[key] = open(path, "a", encoding="utf-8", buffering=1)

        return self._handles[key]

    def _write(self, category: str, record: dict) -> None:
        """Write a record to the appropriate log file."""
        record.setdefault("_ts", datetime.now(timezone.utc).isoformat())
        record.setdefault("_cat", category)
        line = json.dumps(record, default=str) + "\n"

        with self._lock:
            try:
                fh = self._get_handle(category)
                fh.write(line)
                fh.flush()
            except Exception as e:
                logger.error(f"Log write error [{category}]: {e}")

    # ── Trade events ──────────────────────────────────────────────────────────
    def log_trade_open(
        self,
        ticket: int,
        symbol: str,
        strategy: str,
        direction: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        dollar_risk: float,
        signal_confidence: float,
        regime: str,
        slippage_pts: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        self._write("trades", {
            "event": "open",
            "ticket": ticket,
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "dollar_risk": dollar_risk,
            "signal_confidence": signal_confidence,
            "regime": regime,
            "slippage_pts": slippage_pts,
            "latency_ms": latency_ms,
        })

    def log_trade_close(
        self,
        ticket: int,
        symbol: str,
        strategy: str,
        close_price: float,
        realised_pnl: float,
        close_reason: str,
        bars_held: int,
        hours_held: float,
        slippage_pts: float = 0.0,
    ) -> None:
        self._write("trades", {
            "event": "close",
            "ticket": ticket,
            "symbol": symbol,
            "strategy": strategy,
            "close_price": close_price,
            "realised_pnl": realised_pnl,
            "close_reason": close_reason,
            "bars_held": bars_held,
            "hours_held": hours_held,
            "slippage_pts": slippage_pts,
        })

    # ── Signal events ─────────────────────────────────────────────────────────
    def log_signal(
        self,
        strategy: str,
        symbol: str,
        direction: str,
        signal_strength: float,
        z_score: Optional[float] = None,
        trend_strength: Optional[float] = None,
        regime: str = "",
        regime_scale: float = 0.0,
        ic_posterior_mean: float = 0.0,
        actionable: bool = False,
        suspended_reason: Optional[str] = None,
    ) -> None:
        self._write("signals", {
            "strategy": strategy,
            "symbol": symbol,
            "direction": direction,
            "signal_strength": signal_strength,
            "z_score": z_score,
            "trend_strength": trend_strength,
            "regime": regime,
            "regime_scale": regime_scale,
            "ic_posterior_mean": ic_posterior_mean,
            "actionable": actionable,
            "suspended_reason": suspended_reason,
        })

    # ── Risk events ───────────────────────────────────────────────────────────
    def log_risk_event(
        self,
        event_type: str,
        details: dict,
        severity: str = "warning",
    ) -> None:
        self._write("risk", {
            "event_type": event_type,
            "severity": severity,
            **details,
        })

    def log_drawdown(
        self,
        drawdown_pct: float,
        drawdown_level: str,
        daily_loss_pct: float,
        equity: float,
        peak_equity: float,
    ) -> None:
        self._write("risk", {
            "event_type": "drawdown_update",
            "drawdown_pct": drawdown_pct,
            "drawdown_level": drawdown_level,
            "daily_loss_pct": daily_loss_pct,
            "equity": equity,
            "peak_equity": peak_equity,
        })

    def log_kill_switch(self, reason: str, n_positions_closed: int) -> None:
        self._write("risk", {
            "event_type": "kill_switch",
            "severity": "critical",
            "reason": reason,
            "n_positions_closed": n_positions_closed,
        })

    def log_decay_alert(
        self,
        strategy: str,
        instrument: str,
        response: str,
        n_conditions: int,
        active_reasons: list[str],
        metrics: dict,
    ) -> None:
        self._write("risk", {
            "event_type": "decay_alert",
            "strategy": strategy,
            "instrument": instrument,
            "response": response,
            "n_conditions": n_conditions,
            "active_reasons": active_reasons,
            "metrics": metrics,
        })

    # ── System events ─────────────────────────────────────────────────────────
    def log_system(
        self,
        event_type: str,
        details: dict,
        level: str = "info",
    ) -> None:
        self._write("system", {
            "event_type": event_type,
            "level": level,
            **details,
        })

    def log_heartbeat(
        self,
        component_statuses: dict,
        mt5_connected: bool,
        equity: float,
        n_open_positions: int,
        latency_ms: Optional[float] = None,
    ) -> None:
        self._write("system", {
            "event_type": "heartbeat",
            "mt5_connected": mt5_connected,
            "equity": equity,
            "n_open_positions": n_open_positions,
            "latency_ms": latency_ms,
            "components": component_statuses,
        })

    def log_performance(self, snapshot: dict) -> None:
        self._write("system", {
            "event_type": "performance_snapshot",
            **snapshot,
        })

    def log_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        adx: float,
        vol_percentile: float,
        crisis_active: bool,
        strategy_scales: dict,
    ) -> None:
        self._write("system", {
            "event_type": "regime_change",
            "old_regime": old_regime,
            "new_regime": new_regime,
            "adx": adx,
            "vol_percentile": vol_percentile,
            "crisis_active": crisis_active,
            "strategy_scales": strategy_scales,
        })

    def flush_all(self) -> None:
        """Force flush all open file handles."""
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.flush()
                except Exception:
                    pass

    def close(self) -> None:
        """Close all file handles cleanly."""
        with self._lock:
            for fh in self._handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._handles.clear()
