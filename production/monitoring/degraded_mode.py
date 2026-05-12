"""
degraded_mode.py — Degraded Mode Protocol
Implements Section 11.2 Weakness 4 resolution of the framework.

Framework specification (Section 11.2, Weakness 4):
    "System complexity creates fragility. Seven instruments, five alpha engines,
    regime adaptation, Bayesian confidence weighting, dynamic portfolio allocation,
    and a layered monitoring system all have independent failure modes.

    Resolution: Degraded Mode protocol. On ANY system alert, all new entries
    suspended and existing positions managed by stops ONLY. Full complexity
    active only when ALL components are confirmed operational."

Degraded mode levels:
    FULL       — all components operational, normal trading
    DEGRADED_1 — one non-critical component failed, reduced new entries
    DEGRADED_2 — multiple components failed or critical component failed,
                 no new entries, manage existing by stops only
    SAFE_MODE  — kill switch vicinity, trend-only existing positions
    SUSPENDED  — all new entries blocked, manual intervention required

Component criticality:
    CRITICAL : MT5 bridge, risk engine, heartbeat — failure → DEGRADED_2 immediately
    HIGH     : Regime engine, data feed — failure → DEGRADED_1
    MEDIUM   : Portfolio engine, decay monitor — failure → logged, warning only
    LOW      : Performance monitor, structured logger — failure → logged only
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────
class SystemMode(Enum):
    FULL       = "FULL"        # All components operational
    DEGRADED_1 = "DEGRADED_1"  # Reduced new entries (1 non-critical failure)
    DEGRADED_2 = "DEGRADED_2"  # No new entries, manage by stops (critical failure)
    SAFE_MODE  = "SAFE_MODE"   # Trend-only existing, approach kill switch
    SUSPENDED  = "SUSPENDED"   # Full suspension, manual reset required


class ComponentCriticality(Enum):
    CRITICAL = "CRITICAL"   # Failure → DEGRADED_2 immediately
    HIGH     = "HIGH"       # Failure → DEGRADED_1
    MEDIUM   = "MEDIUM"     # Failure → warning only
    LOW      = "LOW"        # Failure → logged only


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
COMPONENT_CRITICALITY: dict[str, ComponentCriticality] = {
    "mt5_bridge":          ComponentCriticality.CRITICAL,
    "risk_engine":         ComponentCriticality.CRITICAL,
    "heartbeat":           ComponentCriticality.CRITICAL,
    "data_feed":           ComponentCriticality.HIGH,
    "regime_engine":       ComponentCriticality.HIGH,
    "signal_engine":       ComponentCriticality.HIGH,
    "portfolio_engine":    ComponentCriticality.MEDIUM,
    "decay_monitor":       ComponentCriticality.MEDIUM,
    "bayesian_estimator":  ComponentCriticality.MEDIUM,
    "performance_monitor": ComponentCriticality.LOW,
    "structured_logger":   ComponentCriticality.LOW,
}


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT FAILURE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ComponentFailure:
    component: str
    criticality: ComponentCriticality
    error: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# DEGRADED MODE MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class DegradedModeManager:
    """
    Monitors component health and manages the system's operating mode.
    Called from the main loop whenever a component fails or recovers.

    In degraded mode the trading loop:
        DEGRADED_1 : Reduce new entry size by 50%, continue all engines
        DEGRADED_2 : No new entries; existing positions managed by stops only
        SAFE_MODE  : Only trend-following positions maintained; no new entries
        SUSPENDED  : All activity stopped; human reset required
    """

    def __init__(self, alert_manager=None):
        self._mode = SystemMode.FULL
        self._failures: dict[str, ComponentFailure] = {}
        self._mode_history: list[dict] = []
        self._alerts = alert_manager

    # ── Failure reporting ─────────────────────────────────────────────────────
    def report_failure(self, component: str, error: str) -> SystemMode:
        """
        Record a component failure and recalculate system mode.
        Returns the new system mode.
        """
        criticality = COMPONENT_CRITICALITY.get(
            component, ComponentCriticality.LOW
        )
        failure = ComponentFailure(
            component=component,
            criticality=criticality,
            error=error,
        )
        self._failures[component] = failure

        old_mode = self._mode
        new_mode = self._calculate_mode()

        if new_mode != old_mode:
            self._transition(old_mode, new_mode, f"component_failure:{component}")

        logger.warning(
            f"Component failure: {component} [{criticality.value}] — "
            f"error='{error}' | mode={new_mode.value}"
        )

        if self._alerts:
            if criticality == ComponentCriticality.CRITICAL:
                self._alerts.critical(
                    f"CRITICAL component failure: {component}\n"
                    f"Error: {error}\nMode: {new_mode.value}"
                )
            else:
                self._alerts.warning(
                    f"Component failure: {component} [{criticality.value}]\n"
                    f"Mode: {new_mode.value}"
                )

        return new_mode

    def report_recovery(self, component: str) -> SystemMode:
        """
        Record a component recovery and recalculate system mode.
        Returns the new system mode.
        """
        if component in self._failures:
            self._failures[component].resolved = True
            self._failures[component].resolved_at = datetime.now(timezone.utc)

        old_mode = self._mode
        new_mode = self._calculate_mode()

        if new_mode != old_mode:
            self._transition(old_mode, new_mode, f"component_recovery:{component}")
            logger.info(
                f"Component recovered: {component} | mode: "
                f"{old_mode.value} → {new_mode.value}"
            )

        return new_mode

    # ── Trading gates ─────────────────────────────────────────────────────────
    def can_open_new_positions(self, strategy: str = "") -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        False if in DEGRADED_2, SAFE_MODE, or SUSPENDED.
        """
        mode = self._mode
        if mode == SystemMode.FULL:
            return True, ""
        if mode == SystemMode.DEGRADED_1:
            return True, "degraded_mode_1_50pct_size"
        if mode == SystemMode.DEGRADED_2:
            return False, "degraded_mode_2_no_new_entries"
        if mode == SystemMode.SAFE_MODE:
            return False, "safe_mode_no_new_entries"
        if mode == SystemMode.SUSPENDED:
            return False, "suspended_manual_reset_required"
        return False, f"unknown_mode_{mode.value}"

    def can_open_trend_position(self) -> tuple[bool, str]:
        """
        In SAFE_MODE, only trend-following positions may continue.
        No new entries of any kind in SUSPENDED.
        """
        mode = self._mode
        if mode in (SystemMode.FULL, SystemMode.DEGRADED_1):
            return True, ""
        if mode == SystemMode.DEGRADED_2:
            return False, "degraded_mode_2"
        if mode == SystemMode.SAFE_MODE:
            return False, "safe_mode_no_new_entries_even_trend"
        return False, f"suspended_or_unknown_{mode.value}"

    def get_size_scale(self) -> float:
        """
        Position size scaling factor for current mode.
        FULL=1.0, DEGRADED_1=0.5, DEGRADED_2/SAFE/SUSPENDED=0.0
        """
        scales = {
            SystemMode.FULL:       1.0,
            SystemMode.DEGRADED_1: 0.5,
            SystemMode.DEGRADED_2: 0.0,
            SystemMode.SAFE_MODE:  0.0,
            SystemMode.SUSPENDED:  0.0,
        }
        return scales.get(self._mode, 0.0)

    def should_manage_by_stops_only(self) -> bool:
        """True if existing positions should only be managed by their stops."""
        return self._mode in (
            SystemMode.DEGRADED_2,
            SystemMode.SAFE_MODE,
            SystemMode.SUSPENDED,
        )

    # ── Mode calculation ──────────────────────────────────────────────────────
    def _calculate_mode(self) -> SystemMode:
        """Recalculate system mode from current failure set."""
        active = {
            comp: f for comp, f in self._failures.items()
            if not f.resolved
        }

        if not active:
            return SystemMode.FULL

        # Any critical failure → DEGRADED_2 minimum
        has_critical = any(
            f.criticality == ComponentCriticality.CRITICAL
            for f in active.values()
        )
        if has_critical:
            return SystemMode.DEGRADED_2

        # Multiple high failures → DEGRADED_2
        high_failures = sum(
            1 for f in active.values()
            if f.criticality == ComponentCriticality.HIGH
        )
        if high_failures >= 2:
            return SystemMode.DEGRADED_2

        # Single high failure → DEGRADED_1
        if high_failures == 1:
            return SystemMode.DEGRADED_1

        # Only medium/low failures → FULL (logged only)
        return SystemMode.FULL

    def _transition(self, old_mode: SystemMode, new_mode: SystemMode, reason: str) -> None:
        """Record mode transition."""
        self._mode = new_mode
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": old_mode.value,
            "to": new_mode.value,
            "reason": reason,
        }
        self._mode_history.append(entry)
        logger.warning(
            f"SYSTEM MODE: {old_mode.value} → {new_mode.value} | reason={reason}"
        )

    def manual_suspend(self, reason: str, operator: str) -> None:
        """Manually suspend the system. Requires named operator."""
        if not operator.strip():
            raise ValueError("Manual suspend requires named operator")
        self._transition(self._mode, SystemMode.SUSPENDED, f"manual:{operator}:{reason}")
        logger.critical(f"System manually suspended by {operator}: {reason}")

    def manual_reset(self, operator: str) -> None:
        """
        Reset from SUSPENDED to FULL after investigation.
        All resolved failures must be cleared first.
        """
        if not operator.strip():
            raise ValueError("Manual reset requires named operator")
        unresolved = [
            c for c, f in self._failures.items() if not f.resolved
        ]
        if unresolved:
            raise RuntimeError(
                f"Cannot reset: unresolved failures remain: {unresolved}. "
                f"Call report_recovery() for each before resetting."
            )
        self._failures.clear()
        self._transition(self._mode, SystemMode.FULL, f"manual_reset:{operator}")
        logger.info(f"System reset to FULL mode by {operator}")

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def mode(self) -> SystemMode:
        return self._mode

    @property
    def is_operational(self) -> bool:
        return self._mode in (SystemMode.FULL, SystemMode.DEGRADED_1)

    @property
    def active_failures(self) -> dict[str, ComponentFailure]:
        return {c: f for c, f in self._failures.items() if not f.resolved}

    def status_summary(self) -> dict:
        return {
            "mode": self._mode.value,
            "is_operational": self.is_operational,
            "can_open_positions": self.can_open_new_positions()[0],
            "size_scale": self.get_size_scale(),
            "manage_by_stops_only": self.should_manage_by_stops_only(),
            "n_active_failures": len(self.active_failures),
            "active_failures": {
                c: {"criticality": f.criticality.value, "error": f.error[:80]}
                for c, f in self.active_failures.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
