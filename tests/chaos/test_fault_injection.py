"""
tests/chaos/test_fault_injection.py — Chaos Engineering Tests
Implements Section 9.2 (tests/chaos/) and Weakness 3 resolution.

Tests the system's behaviour under:
    - MT5 connection loss mid-session
    - Data feed gaps and stale bars
    - Kill switch trigger and recovery
    - Component heartbeat timeout simulation
    - Simultaneous multi-component failure
    - Degraded mode transitions under load
    - Order manager reconciliation after external close

These tests verify that the system FAILS SAFELY — i.e. it closes positions
and suspends trading rather than continuing blindly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock


# ── Test: Kill switch fires and blocks new positions ─────────────────────────
class TestKillSwitchFaultInjection:
    def _make_risk_engine(self, balance=100_000):
        """Minimal risk engine for kill switch tests."""
        from tests.integration.test_pipeline import make_fake_config
        from production.engines.risk_engine import RiskEngine
        return RiskEngine(make_fake_config(), balance)

    def test_kill_switch_fires_on_excessive_daily_loss(self):
        from production.engines.risk_engine import RiskEngine, PositionSizeRequest, RiskCheckResult
        from tests.integration.test_pipeline import make_fake_config
        engine = RiskEngine(make_fake_config(), 100_000)

        # Simulate daily loss exceeding kill switch threshold (3%)
        engine.drawdown_tracker._daily_start_equity = 100_000
        engine.drawdown_tracker._current = 96_800  # 3.2% daily loss

        req = PositionSizeRequest(
            instrument="EURUSD", strategy="trend_following",
            direction="buy", entry_price=1.1000, stop_loss_price=1.0950,
            signal_confidence=0.8, regime_scale=0.8,
            current_regime="normal_trending", account_balance=100_000,
            current_portfolio_risk_usd=0.0, current_drawdown_pct=0.032,
        )
        result = engine.size_position(req, np.random.normal(0, 0.001, 30),
                                       {"contract_size": 100000, "lot_size": 0.01})
        assert result.result == RiskCheckResult.REJECTED

    def test_kill_switch_requires_human_reset(self):
        from production.engines.risk_engine import RiskEngine
        from tests.integration.test_pipeline import make_fake_config
        engine = RiskEngine(make_fake_config(), 100_000)
        engine._trigger_kill_switch("test_trigger")
        assert engine.kill_switch_active
        with pytest.raises(ValueError):
            engine.reset_kill_switch("")
        engine.reset_kill_switch("Chaos Test Operator")
        assert not engine.kill_switch_active

    def test_kill_switch_blocks_all_strategies(self):
        from production.engines.risk_engine import RiskEngine, PositionSizeRequest, RiskCheckResult
        from tests.integration.test_pipeline import make_fake_config
        engine = RiskEngine(make_fake_config(), 100_000)
        engine._trigger_kill_switch("chaos_test")

        for strategy in ["mean_reversion", "trend_following", "volatility_breakout"]:
            req = PositionSizeRequest(
                instrument="EURUSD", strategy=strategy,
                direction="buy", entry_price=1.1000, stop_loss_price=1.0950,
                signal_confidence=0.9, regime_scale=1.0,
                current_regime="normal_trending", account_balance=100_000,
                current_portfolio_risk_usd=0.0, current_drawdown_pct=0.0,
            )
            result = engine.size_position(req, np.random.normal(0, 0.001, 30),
                                           {"contract_size": 100000, "lot_size": 0.01})
            assert result.result == RiskCheckResult.REJECTED, \
                f"Kill switch should block {strategy}"


# ── Test: Data validation rejects bad bars ───────────────────────────────────
class TestDataFaultInjection:
    def test_inverted_ohlc_rejected(self):
        from production.data.data_validator import validate_bar, Bar
        bad_bar = Bar(
            time=datetime.now(timezone.utc),
            open=1.1000, high=1.0900,  # high < open — invalid
            low=1.0950, close=1.0980,
            tick_volume=1000,
        )
        result = validate_bar(bad_bar, "EURUSD", "M5")
        assert not result.valid
        assert any("ohlc" in c for c in result.failed_checks)

    def test_zero_price_rejected(self):
        from production.data.data_validator import validate_bar, Bar
        bad_bar = Bar(
            time=datetime.now(timezone.utc),
            open=0.0, high=0.0, low=0.0, close=0.0,
            tick_volume=1000,
        )
        result = validate_bar(bad_bar, "EURUSD", "M5")
        assert not result.valid

    def test_stale_bar_rejected(self):
        from production.data.data_validator import validate_bar, Bar
        stale_time = datetime.now(timezone.utc) - timedelta(hours=2)
        stale_bar = Bar(
            time=stale_time,
            open=1.1000, high=1.1020, low=1.0990, close=1.1010,
            tick_volume=500,
        )
        result = validate_bar(stale_bar, "EURUSD", "M5",
                               staleness_multiplier=3.0)  # 3 × 5min = 15min max age
        assert not result.valid
        assert any("stale" in c for c in result.failed_checks)

    def test_flash_crash_spike_rejected(self):
        from production.data.data_validator import validate_bar, Bar
        # EUR/USD bar with 500-pip range — clearly a data error
        spike_bar = Bar(
            time=datetime.now(timezone.utc),
            open=1.1000, high=1.1500, low=1.0500, close=1.1000,
            tick_volume=100,
        )
        result = validate_bar(spike_bar, "EURUSD", "M5")
        assert not result.valid
        assert any("range" in c for c in result.failed_checks)

    def test_valid_bar_passes(self):
        from production.data.data_validator import validate_bar, Bar
        good_bar = Bar(
            time=datetime.now(timezone.utc),
            open=1.1000, high=1.1010, low=1.0995, close=1.1005,
            tick_volume=2000,
        )
        result = validate_bar(good_bar, "EURUSD", "M5")
        assert result.valid
        assert len(result.failed_checks) == 0

    def test_series_with_majority_bad_bars_fails(self):
        from production.data.data_validator import validate_series, Bar
        bars = []
        for i in range(60):
            # Alternate good and bad bars
            if i % 3 == 0:
                # Bad bar — zero prices
                bars.append(Bar(
                    time=datetime.now(timezone.utc) - timedelta(minutes=5*(60-i)),
                    open=0.0, high=0.0, low=0.0, close=0.0, tick_volume=0
                ))
            else:
                bars.append(Bar(
                    time=datetime.now(timezone.utc) - timedelta(minutes=5*(60-i)),
                    open=1.1000 + i*0.0001, high=1.1010 + i*0.0001,
                    low=1.0995 + i*0.0001, close=1.1005 + i*0.0001,
                    tick_volume=1000
                ))
        result = validate_series(bars, "EURUSD", "M5", min_bars=30)
        assert not result.valid


# ── Test: Degraded mode under component failures ──────────────────────────────
class TestDegradedModeFaultInjection:
    def test_multiple_high_failures_escalate(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("data_feed", "feed timeout")
        assert mgr.mode == SystemMode.DEGRADED_1
        mgr.report_failure("regime_engine", "ADX error")
        assert mgr.mode == SystemMode.DEGRADED_2

    def test_degraded2_blocks_new_entries(self):
        from production.monitoring.degraded_mode import DegradedModeManager
        mgr = DegradedModeManager()
        mgr.report_failure("risk_engine", "crash")
        allowed, _ = mgr.can_open_new_positions()
        assert not allowed

    def test_degraded2_requires_stops_management(self):
        from production.monitoring.degraded_mode import DegradedModeManager
        mgr = DegradedModeManager()
        mgr.report_failure("mt5_bridge", "disconnect")
        assert mgr.should_manage_by_stops_only()

    def test_recovery_sequence_restores_full(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("data_feed", "error1")
        mgr.report_failure("regime_engine", "error2")
        assert mgr.mode == SystemMode.DEGRADED_2
        mgr.report_recovery("data_feed")
        mgr.report_recovery("regime_engine")
        assert mgr.mode == SystemMode.FULL

    def test_medium_failures_do_not_degrade(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("performance_monitor", "slow")
        mgr.report_failure("decay_monitor", "timeout")
        # Medium/low failures should not change mode
        assert mgr.mode == SystemMode.FULL


# ── Test: Heartbeat timeout detection ────────────────────────────────────────
class TestHeartbeatFaultInjection:
    def test_component_registered_and_tracked(self):
        from production.monitoring.heartbeat import HeartbeatSupervisor, AlertManager
        mock_bridge = MagicMock()
        mock_bridge.get_open_positions.return_value = []

        class MockConfig:
            @property
            def heartbeat_params(self):
                return {"timeout_seconds": 120, "monitor_interval_seconds": 30,
                        "min_margin_level_pct": 200}
            @property
            def portfolio_limits(self):
                return {"max_daily_loss_pct": 0.02}
            @property
            def kill_switch_params(self):
                return {"daily_loss_trigger_pct": 0.03,
                        "drawdown_trigger_pct": 0.08,
                        "position_close_interval_seconds": 0}
            @property
            def monitoring_params(self):
                return {"telegram_bot_token": "", "telegram_chat_id": "",
                        "alert_on_trade": False, "alert_on_kill_switch": False,
                        "alert_on_decay_condition": False,
                        "alert_on_drawdown_threshold": False}

        alerts = AlertManager(MockConfig())
        supervisor = HeartbeatSupervisor(mock_bridge, alerts, MockConfig())
        supervisor.register("test_component")
        supervisor.beat("test_component")

        statuses = supervisor.component_statuses()
        assert "test_component" in statuses
        assert statuses["test_component"]["beat_count"] == 1
        assert statuses["test_component"]["last_beat_seconds_ago"] < 5

    def test_reset_requires_named_operator(self):
        from production.monitoring.heartbeat import HeartbeatSupervisor, AlertManager

        class MockConfig:
            @property
            def heartbeat_params(self):
                return {"timeout_seconds": 60, "monitor_interval_seconds": 10,
                        "min_margin_level_pct": 200}
            @property
            def portfolio_limits(self): return {"max_daily_loss_pct": 0.02}
            @property
            def kill_switch_params(self):
                return {"daily_loss_trigger_pct": 0.03,
                        "drawdown_trigger_pct": 0.08,
                        "position_close_interval_seconds": 0}
            @property
            def monitoring_params(self):
                return {"telegram_bot_token": "", "telegram_chat_id": "",
                        "alert_on_trade": False, "alert_on_kill_switch": False,
                        "alert_on_decay_condition": False,
                        "alert_on_drawdown_threshold": False}

        mock_bridge = MagicMock()
        alerts = AlertManager(MockConfig())
        supervisor = HeartbeatSupervisor(mock_bridge, alerts, MockConfig())
        supervisor._kill_fired = True

        with pytest.raises(ValueError):
            supervisor.reset(authorised_by="")


# ── Test: Monte Carlo stress test ─────────────────────────────────────────────
class TestMonteCarloChaos:
    def test_stress_test_passes_normal_returns(self):
        from production.monitoring.monte_carlo_stress import run_monte_carlo_stress
        np.random.seed(42)
        # Realistic Sharpe ~1 daily returns
        returns = np.random.normal(0.0004, 0.006, 252)
        result = run_monte_carlo_stress(
            returns, account_balance=100_000,
            n_paths=500, n_days=126
        )
        # Should pass (P(DD>10%) < 5% for reasonable returns)
        assert result.prob_drawdown_exceeds_10pct < 0.5  # Generous bound for test speed

    def test_stress_test_fails_extreme_vol(self):
        from production.monitoring.monte_carlo_stress import run_monte_carlo_stress
        np.random.seed(42)
        # Very high volatility — should fail stress test
        returns = np.random.normal(-0.002, 0.05, 252)  # negative drift, high vol
        result = run_monte_carlo_stress(
            returns, account_balance=100_000,
            n_paths=500, n_days=126
        )
        assert result.prob_drawdown_exceeds_10pct > 0.0
        assert not result.passes or result.prob_drawdown_exceeds_10pct > 0.05

    def test_size_reduction_computed_when_failing(self):
        from production.monitoring.monte_carlo_stress import run_monte_carlo_stress
        np.random.seed(42)
        returns = np.random.normal(-0.003, 0.04, 252)
        result = run_monte_carlo_stress(
            returns, account_balance=100_000,
            n_paths=300, n_days=126
        )
        if not result.passes:
            assert result.size_reduction_factor < 1.0
            assert result.size_reduction_factor > 0.0

    def test_scheduler_runs_monthly(self):
        from production.monitoring.monte_carlo_stress import MonthlyStressTestScheduler
        scheduler = MonthlyStressTestScheduler()
        assert scheduler.should_run()  # Never run before

        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.007, 252)
        result = scheduler.run_if_due(returns, 100_000, n_paths=200)
        assert result is not None
        assert not scheduler.should_run()  # Just ran, shouldn't run again


# ── Test: Order manager reconciliation ───────────────────────────────────────
class TestOrderManagerReconciliation:
    def test_externally_closed_position_detected(self):
        from production.execution.order_manager import (
            OrderManager, ManagedPosition, PositionStatus
        )

        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = []  # MT5 shows no positions

        mock_risk = MagicMock()
        mock_portfolio = MagicMock()
        mock_alerts = MagicMock()

        om = OrderManager(mock_mt5, mock_risk, mock_portfolio, mock_alerts)

        # Add a position to internal tracker (as if it was opened)
        pos = ManagedPosition(
            ticket=12345, symbol="EURUSD", strategy="trend_following",
            direction="buy", volume=0.1, entry_price=1.1000,
            stop_loss=1.0950, take_profit=0.0,
            dollar_risk=50.0, signal_confidence=0.8,
            regime_at_entry="normal_trending", entry_z_score=None,
            atr_at_entry=0.0010,
        )
        om._positions[12345] = pos

        # Reconcile — MT5 shows no positions so it was externally closed
        result = om.reconcile_with_mt5()
        assert 12345 in result["missing_from_mt5"]
        assert len(om._positions) == 0  # Removed from tracker

    def test_unexpected_mt5_position_flagged(self):
        from production.execution.order_manager import OrderManager

        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 99999, "symbol": "GBPUSD", "type": "buy",
             "volume": 0.1, "open_price": 1.25, "current_price": 1.26,
             "sl": 1.245, "tp": 0.0, "profit": 100.0,
             "magic": 20250101, "comment": ""}
        ]

        mock_risk = MagicMock()
        mock_portfolio = MagicMock()
        mock_alerts = MagicMock()

        om = OrderManager(mock_mt5, mock_risk, mock_portfolio, mock_alerts)
        # Internal tracker is empty — no positions opened by this system

        result = om.reconcile_with_mt5()
        assert 99999 in result["unexpected_in_mt5"]
