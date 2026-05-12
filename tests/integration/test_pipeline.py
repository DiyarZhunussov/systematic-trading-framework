"""
tests/integration/test_pipeline.py — Integration Tests
Tests the full pipeline: config → data → regime → signal → risk sizing.
Runs without MT5 (uses synthetic data).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ── Shared helpers ────────────────────────────────────────────────────────────
def make_bar_array(n=300, trend=0.0002):
    """Create a realistic synthetic BarArray."""
    np.random.seed(99)
    prices = np.cumprod(1 + np.random.normal(trend, 0.008, n)) * 1.1000
    high   = prices * np.random.uniform(1.000, 1.005, n)
    low    = prices * np.random.uniform(0.995, 1.000, n)
    volume = np.random.randint(500, 5000, n)
    now    = datetime.now(timezone.utc)
    timestamps = [now - timedelta(minutes=5*(n-i)) for i in range(n)]

    class BA:
        pass
    ba = BA()
    ba.close = prices
    ba.open  = prices * np.random.uniform(0.999, 1.001, n)
    ba.high  = high
    ba.low   = low
    ba.volume = volume
    ba.timestamps = timestamps
    ba.n = n
    ba.returns = lambda log=True: (
        np.diff(np.log(prices + 1e-10)) if log
        else np.diff(prices) / (prices[:-1] + 1e-10)
    )
    return ba


def make_fake_config():
    """Minimal config object for integration tests."""
    class FakeCfg:
        risk = {
            "trade": {"max_risk_per_trade_pct": 0.005,
                      "max_stop_distance_atr_multiple": 1.5,
                      "max_spread_multiple": 3.0,
                      "slippage_budget_pips": 2.0},
            "strategy": {"max_daily_loss_pct": 0.015,
                         "max_concurrent_trades": 3,
                         "usd_net_exposure_max_pct": 0.03,
                         "equity_index_net_exposure_max_pct": 0.05,
                         "mean_reversion_max_risk_budget_pct": 0.15,
                         "breakout_max_risk_budget_pct": 0.10,
                         "stat_arb_allocation_pct": 0.0},
            "portfolio": {"max_daily_loss_pct": 0.020,
                          "max_weekly_loss_pct": 0.040,
                          "drawdown_review_trigger_pct": 0.060,
                          "drawdown_suspension_trigger_pct": 0.080,
                          "drawdown_hard_limit_pct": 0.100,
                          "extended_drawdown_days": 10,
                          "extended_drawdown_threshold_pct": 0.030,
                          "max_gross_leverage": 5.0,
                          "max_aggregate_strategy_scale": 1.0,
                          "min_free_margin_pct": 0.30,
                          "max_cvar_5pct_daily_pct": 0.030},
            "kill_switch": {"daily_loss_trigger_pct": 0.030,
                            "drawdown_trigger_pct": 0.080,
                            "margin_level_trigger_pct": 150,
                            "position_close_interval_seconds": 0,
                            "requires_manual_reset": True},
            "volatility_target": {"normal_annualised_pct": 0.10,
                                  "high_vol_annualised_pct": 0.07,
                                  "crisis_annualised_pct": 0.04,
                                  "smoothing_ema_days": 3},
            "kelly": {"fraction": 0.25, "min_fraction": 0.20, "max_fraction": 0.40},
            "decay_monitor": {"rolling_sharpe_warn_threshold": 0.50,
                              "rolling_ic_negative_days": 10,
                              "hostile_regime_days": 20,
                              "slippage_deterioration_pct": 0.20,
                              "cusum_alert_sigma": 3.0,
                              "response_1_condition_allocation_pct": 0.75,
                              "response_2_condition_allocation_pct": 0.50,
                              "response_3_condition_suspend": True,
                              "response_4plus_emergency_retire": True},
            "heartbeat": {"timeout_seconds": 120,
                          "monitor_interval_seconds": 30,
                          "min_margin_level_pct": 200},
            "drawdown_response": [
                {"threshold_pct": 0.03, "scale_factor": 1.00, "action": "normal"},
                {"threshold_pct": 0.05, "scale_factor": 0.75, "action": "reduce_monitor"},
                {"threshold_pct": 0.07, "scale_factor": 0.50, "action": "suspend_mean_rev"},
                {"threshold_pct": 0.09, "scale_factor": 0.25, "action": "trend_only"},
                {"threshold_pct": 0.10, "scale_factor": 0.00, "action": "full_suspension"},
            ],
        }
        @property
        def trade_limits(self): return self.risk["trade"]
        @property
        def strategy_limits(self): return self.risk["strategy"]
        @property
        def portfolio_limits(self): return self.risk["portfolio"]
        @property
        def kill_switch_params(self): return self.risk["kill_switch"]
        @property
        def vol_target_params(self): return self.risk["volatility_target"]
        @property
        def kelly_params(self): return self.risk["kelly"]
        @property
        def decay_params(self): return self.risk["decay_monitor"]
        @property
        def heartbeat_params(self): return self.risk["heartbeat"]
        @property
        def bayesian_params(self):
            return {"ic_prior_alpha": 2, "ic_prior_beta": 30, "uncertainty_aversion": 0.5}
        @property
        def regime_params(self):
            return {"adx_trending_threshold": 25.0, "adx_choppy_threshold": 20.0,
                    "adx_period": 14, "vol_percentile_low": 25, "vol_percentile_high": 75,
                    "vol_percentile_crisis": 90, "vol_lookback_days": 252,
                    "regime_change_confirmation_days": 1,
                    "crisis_equity_vol_ratio": 2.5, "crisis_fx_vol_zscore": 2.0,
                    "crisis_gold_return_sigma": 3.0, "crisis_min_indicators": 2,
                    "crisis_resolution_days": 3}
        def instrument(self, sym):
            return {"mean_reversion_threshold": 2.0, "mean_reversion_window": 20,
                    "atr_period": 14, "time_stop_hours": 3,
                    "trend_fast_ema": 10, "trend_slow_ema": 50,
                    "donchian_period_daily": 20, "donchian_period_4h": 40,
                    "breakout_confirm_bars": 3, "breakout_lookback_bars": 8,
                    "contract_size": 100000, "lot_size": 0.01,
                    "point_value": 0.0001, "asset_class": "forex"}
        @property
        def instruments(self): return {"instruments": {}}
    return FakeCfg()


# ── Test: Regime detection pipeline ──────────────────────────────────────────
class TestRegimePipeline:
    def test_regime_detects_from_bar_array(self):
        from production.engines.regime_engine import RegimeEngine, VolatilityLevel
        cfg = make_fake_config()
        engine = RegimeEngine(cfg)
        bars = make_bar_array(n=300)
        regime = engine.compute_regime(bars)
        assert regime is not None
        assert isinstance(regime.vol_level, VolatilityLevel)

    def test_regime_scales_normalised(self):
        from production.engines.regime_engine import RegimeEngine
        cfg = make_fake_config()
        engine = RegimeEngine(cfg)
        bars = make_bar_array(n=300)
        regime = engine.compute_regime(bars)
        scales = engine.get_strategy_scales(regime)
        total = scales.trend_following + scales.mean_reversion + scales.volatility_breakout
        assert total <= 1.0 + 1e-9

    def test_crisis_detection_inactive_in_normal_market(self):
        from production.engines.regime_engine import RegimeEngine
        cfg = make_fake_config()
        engine = RegimeEngine(cfg)
        bars = make_bar_array(n=300)
        regime = engine.compute_regime(bars)
        # Normal synthetic data should not trigger crisis
        assert regime.crisis_factors_active < 2


# ── Test: Signal → Risk sizing pipeline ──────────────────────────────────────
class TestSignalToSizingPipeline:
    def test_mean_reversion_signal_to_size(self):
        from production.engines.signal_engine_mean_reversion import MeanReversionEngine, SignalDirection
        from production.engines.risk_engine import RiskEngine, PositionSizeRequest, RiskCheckResult
        cfg = make_fake_config()
        bars = make_bar_array(n=300)

        mr_engine = MeanReversionEngine(cfg)
        signal = mr_engine.compute_signal(bars, "EURUSD", regime_scale=0.8)

        risk_engine = RiskEngine(cfg, 100_000)
        req = PositionSizeRequest(
            instrument="EURUSD", strategy="mean_reversion",
            direction="buy" if signal.direction == SignalDirection.LONG else "sell",
            entry_price=float(bars.close[-1]),
            stop_loss_price=float(bars.close[-1]) - 0.0050,
            signal_confidence=0.7, regime_scale=0.8,
            current_regime="normal_choppy",
            account_balance=100_000,
            current_portfolio_risk_usd=0.0,
            current_drawdown_pct=0.0,
        )
        result = risk_engine.size_position(req, bars.returns(), cfg.instrument("EURUSD"))
        # Either approved (if signal was actionable) or rejected (if not) — both valid
        assert result.result in list(RiskCheckResult)

    def test_trend_following_signal_to_size(self):
        from production.engines.signal_engine_trend_following import TrendFollowingEngine, SignalDirection
        from production.engines.risk_engine import RiskEngine, PositionSizeRequest
        cfg = make_fake_config()
        bars = make_bar_array(n=300, trend=0.0005)  # Strong uptrend

        tf_engine = TrendFollowingEngine(cfg)
        signal = tf_engine.compute_signal(bars, "EURUSD", timeframe="D1",
                                           regime_scale=1.0, is_daily=True)

        risk_engine = RiskEngine(cfg, 100_000)
        req = PositionSizeRequest(
            instrument="EURUSD", strategy="trend_following",
            direction="buy", entry_price=float(bars.close[-1]),
            stop_loss_price=float(bars.close[-1]) - 0.0100,
            signal_confidence=0.6, regime_scale=1.0,
            current_regime="normal_trending",
            account_balance=100_000,
            current_portfolio_risk_usd=0.0,
            current_drawdown_pct=0.0,
        )
        result = risk_engine.size_position(req, bars.returns(), cfg.instrument("EURUSD"))
        assert result is not None
        assert result.position_size_lots >= 0.0


# ── Test: Bayesian → confidence → sizing ─────────────────────────────────────
class TestBayesianToSizing:
    def test_confidence_weight_reduces_size(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        from production.engines.risk_engine import RiskEngine, PositionSizeRequest

        cfg = make_fake_config()
        bars = make_bar_array(n=100)
        risk_engine = RiskEngine(cfg, 100_000)

        # High confidence (many wins)
        est_high = ICBayesianEstimator(2, 30)
        est_high.update(80, 100)
        w_high = est_high.confidence_weight(0.04)

        # Low confidence (many losses)
        est_low = ICBayesianEstimator(2, 30)
        est_low.update(10, 100)
        w_low = est_low.confidence_weight(0.04)

        assert w_high > w_low

        def make_req(conf):
            return PositionSizeRequest(
                instrument="EURUSD", strategy="mean_reversion",
                direction="buy", entry_price=1.1000,
                stop_loss_price=1.0950,
                signal_confidence=conf, regime_scale=0.8,
                current_regime="normal_choppy",
                account_balance=100_000,
                current_portfolio_risk_usd=0.0,
                current_drawdown_pct=0.0,
            )

        instr_cfg = {"contract_size": 100000, "lot_size": 0.01}
        r_high = risk_engine.size_position(make_req(w_high), bars.returns(), instr_cfg)
        r_low  = risk_engine.size_position(make_req(w_low),  bars.returns(), instr_cfg)

        if r_high.is_approved and r_low.is_approved:
            assert r_high.position_size_lots >= r_low.position_size_lots


# ── Test: Portfolio engine pipeline ──────────────────────────────────────────
class TestPortfolioPipeline:
    def test_rebalance_produces_valid_weights(self):
        from production.engines.portfolio_engine import PortfolioEngine
        from production.engines.portfolio_engine import equal_risk_contribution_weights

        class PCfg:
            @property
            def portfolio_params(self):
                return {"rebalance_frequency_days": 30, "erc_vol_lookback_days": 60,
                        "ewma_decay": 0.94, "min_strategy_weight": 0.05,
                        "max_strategy_weight": 0.60, "weight_change_threshold_pct": 0.10,
                        "diversification_ratio_warn": 1.20}

        engine = PortfolioEngine(PCfg())

        # Add some return history
        np.random.seed(42)
        for _ in range(80):
            engine.record_returns({
                "mean_reversion": np.random.normal(0.0003, 0.008),
                "trend_following": np.random.normal(0.0002, 0.010),
                "volatility_breakout": np.random.normal(0.0001, 0.009),
            })

        allocation = engine.rebalance(
            strategy_volatilities={"mean_reversion": 0.12,
                                   "trend_following": 0.15,
                                   "volatility_breakout": 0.13},
            regime_scales={"mean_reversion": 0.6,
                           "trend_following": 0.8,
                           "volatility_breakout": 0.4},
            confidence_weights={"mean_reversion": 0.7,
                                 "trend_following": 0.8,
                                 "volatility_breakout": 0.5},
            account_balance=100_000,
        )

        total = sum(allocation.weights.values())
        assert abs(total - 1.0) < 1e-6
        for w in allocation.weights.values():
            assert 0.0 <= w <= 1.0


# ── Test: Decay monitor pipeline ─────────────────────────────────────────────
class TestDecayMonitorPipeline:
    def test_normal_operations_no_alert(self):
        from production.monitoring.decay_monitor import AlphaDecayMonitor, DecayResponse

        class DCfg:
            @property
            def decay_params(self):
                return {"rolling_sharpe_warn_threshold": 0.50,
                        "rolling_ic_negative_days": 10,
                        "hostile_regime_days": 20,
                        "slippage_deterioration_pct": 0.20,
                        "cusum_alert_sigma": 3.0,
                        "response_1_condition_allocation_pct": 0.75,
                        "response_2_condition_allocation_pct": 0.50,
                        "response_3_condition_suspend": True,
                        "response_4plus_emergency_retire": True}

        monitor = AlphaDecayMonitor(DCfg())
        monitor.set_historical_sharpe("trend_following", "EURUSD", 1.2)

        # Feed healthy P&L
        np.random.seed(42)
        for _ in range(90):
            monitor.record_daily_pnl("trend_following", "EURUSD",
                                     float(np.random.normal(50, 30)))
            monitor.record_signal_strength("trend_following", "EURUSD",
                                           float(np.random.uniform(0.3, 0.8)))

        metrics = monitor.compute_metrics("trend_following", "EURUSD")
        assert metrics.response == DecayResponse.NORMAL

    def test_negative_pnl_triggers_alert(self):
        from production.monitoring.decay_monitor import AlphaDecayMonitor, DecayResponse

        class DCfg:
            @property
            def decay_params(self):
                return {"rolling_sharpe_warn_threshold": 0.50,
                        "rolling_ic_negative_days": 10,
                        "hostile_regime_days": 20,
                        "slippage_deterioration_pct": 0.20,
                        "cusum_alert_sigma": 3.0,
                        "response_1_condition_allocation_pct": 0.75,
                        "response_2_condition_allocation_pct": 0.50,
                        "response_3_condition_suspend": True,
                        "response_4plus_emergency_retire": True}

        monitor = AlphaDecayMonitor(DCfg())
        monitor.set_historical_sharpe("mean_reversion", "GBPUSD", 1.5)

        # Feed very negative P&L (structural decay)
        np.random.seed(42)
        for _ in range(90):
            monitor.record_daily_pnl("mean_reversion", "GBPUSD",
                                     float(np.random.normal(-100, 30)))

        metrics = monitor.compute_metrics("mean_reversion", "GBPUSD")
        assert metrics.response != DecayResponse.NORMAL


# ── Test: Degraded mode manager ───────────────────────────────────────────────
class TestDegradedMode:
    def test_full_mode_by_default(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        assert mgr.mode == SystemMode.FULL

    def test_critical_failure_goes_to_degraded2(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("mt5_bridge", "connection timeout")
        assert mgr.mode == SystemMode.DEGRADED_2

    def test_high_failure_goes_to_degraded1(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("regime_engine", "ADX computation error")
        assert mgr.mode == SystemMode.DEGRADED_1

    def test_recovery_restores_full_mode(self):
        from production.monitoring.degraded_mode import DegradedModeManager, SystemMode
        mgr = DegradedModeManager()
        mgr.report_failure("regime_engine", "error")
        assert mgr.mode == SystemMode.DEGRADED_1
        mgr.report_recovery("regime_engine")
        assert mgr.mode == SystemMode.FULL

    def test_no_new_entries_in_degraded2(self):
        from production.monitoring.degraded_mode import DegradedModeManager
        mgr = DegradedModeManager()
        mgr.report_failure("risk_engine", "sizing failed")
        allowed, reason = mgr.can_open_new_positions()
        assert not allowed
        assert reason != ""

    def test_size_scale_degraded1(self):
        from production.monitoring.degraded_mode import DegradedModeManager
        mgr = DegradedModeManager()
        mgr.report_failure("data_feed", "stale data")
        assert mgr.get_size_scale() == 0.5

    def test_manual_reset_requires_name(self):
        from production.monitoring.degraded_mode import DegradedModeManager
        mgr = DegradedModeManager()
        mgr.report_failure("regime_engine", "error")
        mgr.report_recovery("regime_engine")
        with pytest.raises(ValueError):
            mgr.manual_reset("")
        mgr.manual_reset("Test Operator")


# ── Test: Validation tools ────────────────────────────────────────────────────
class TestValidationTools:
    def test_dsr_example_from_pdf(self):
        from research.validation.deflated_sharpe import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(
            sr_observed=0.80, t=756, skew=0.15, kurt=1.20, n_trials=50
        )
        assert 0.0 <= result.dsr <= 1.0
        assert result.sr_benchmark > 0

    def test_dsr_higher_n_trials_lower_dsr(self):
        from research.validation.deflated_sharpe import deflated_sharpe_ratio
        r10  = deflated_sharpe_ratio(0.80, 756, 0.15, 1.20, n_trials=10)
        r100 = deflated_sharpe_ratio(0.80, 756, 0.15, 1.20, n_trials=100)
        assert r10.dsr > r100.dsr

    def test_bh_finds_genuinely_significant(self):
        from research.validation.multiple_testing import benjamini_hochberg
        # Two genuine, many noise
        p_vals = list(np.random.uniform(0.1, 1.0, 18)) + [0.001, 0.003]
        names  = [f"noise_{i}" for i in range(18)] + ["genuine_1", "genuine_2"]
        result = benjamini_hochberg(p_vals, names, fdr_target=0.05)
        assert "genuine_1" in result.significant_strategies()
        assert "genuine_2" in result.significant_strategies()

    def test_bonferroni_more_conservative_than_bh(self):
        from research.validation.multiple_testing import (
            benjamini_hochberg, bonferroni_correction
        )
        np.random.seed(42)
        p_vals = list(np.random.uniform(0.01, 0.10, 20))
        bh   = benjamini_hochberg(p_vals)
        bonf = bonferroni_correction(p_vals)
        assert bonf.n_rejected <= bh.n_rejected

    def test_pbo_noise_near_half(self):
        from research.validation.pbo import probability_of_backtest_overfitting
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, (500, 10))
        result = probability_of_backtest_overfitting(
            noise, n_subsamples=8, n_combinations=200
        )
        # Pure noise PBO should be close to 0.5
        assert 0.2 <= result.pbo <= 0.8

    def test_opportunity_ranker_framework_scores(self):
        from research.validation.opportunity_ranker import (
            OpportunityRanker, get_framework_scores, RESEARCH_THRESHOLD
        )
        ranker = OpportunityRanker()
        for opp in get_framework_scores():
            ranker.add(opp)
        # Trend following must be top ranked
        ranked = ranker.ranked()
        assert ranked[0].name == "Trend Following"
        # Stat arb must NOT meet threshold
        stat_arb = next(o for o in ranked if o.name == "Statistical Arbitrage")
        assert not stat_arb.warrants_research
        # Trend, regime, MR, breakout must meet threshold
        eligible_names = {o.name for o in ranker.eligible_for_research()}
        assert "Trend Following" in eligible_names
        assert "Statistical Arbitrage" not in eligible_names

    def test_walk_forward_fold_generation(self):
        from research.validation.walk_forward import generate_walk_forward_folds
        folds = generate_walk_forward_folds(
            n_observations=756, n_folds=4,
            embargo_periods=10, min_train_periods=100
        )
        assert len(folds) == 4
        for fold in folds:
            assert fold.train_end > fold.train_start
            assert fold.test_end > fold.test_start
            assert fold.embargo_end > fold.embargo_start
            # No overlap between train and test
            assert fold.test_start >= fold.embargo_end

    def test_parameter_count_check(self):
        from research.validation.anti_overfitting_checklist import check_parameter_count
        assert check_parameter_count(8, 80)["passes"]
        assert not check_parameter_count(9, 80)["passes"]
        assert check_parameter_count(1, 80)["passes"]
