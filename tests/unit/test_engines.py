"""
tests/unit/test_engines.py — Unit Tests for Alpha Engines and Risk Engine
Covers: signal engines, regime engine, Bayesian estimator, risk engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from datetime import datetime, timezone

# ── Shared fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_prices():
    np.random.seed(42)
    return np.cumprod(1 + np.random.normal(0.0002, 0.01, 500))

@pytest.fixture
def sample_bar_array(sample_prices):
    """Minimal BarArray-like object for testing."""
    class FakeBarArray:
        def __init__(self, prices):
            n = len(prices)
            self.close = prices
            self.open  = prices * np.random.uniform(0.999, 1.001, n)
            self.high  = prices * np.random.uniform(1.000, 1.005, n)
            self.low   = prices * np.random.uniform(0.995, 1.000, n)
            self.volume = np.random.randint(100, 10000, n)
            now = datetime.now(timezone.utc)
            from datetime import timedelta
            self.timestamps = [now - timedelta(minutes=5*(n-i)) for i in range(n)]
            self.n = n
        def returns(self, log=True):
            if log:
                return np.diff(np.log(self.close + 1e-10))
            return np.diff(self.close) / (self.close[:-1] + 1e-10)
    return FakeBarArray(sample_prices)


# ── Mean Reversion Engine ─────────────────────────────────────────────────────
class TestMeanReversionIndicators:
    def test_rolling_zscore_length(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import rolling_zscore
        z = rolling_zscore(sample_prices, window=20)
        assert len(z) == len(sample_prices)

    def test_rolling_zscore_nan_prefix(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import rolling_zscore
        z = rolling_zscore(sample_prices, window=20)
        assert np.all(np.isnan(z[:20]))
        assert not np.isnan(z[20])

    def test_rolling_zscore_mean_zero(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import rolling_zscore
        z = rolling_zscore(sample_prices, window=20)
        valid = z[~np.isnan(z)]
        assert abs(np.mean(valid)) < 0.5  # Should be roughly mean-zero

    def test_autocorrelation_length(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import rolling_autocorrelation
        rho = rolling_autocorrelation(sample_prices, window=30)
        assert len(rho) == len(sample_prices)

    def test_autocorrelation_bounds(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import rolling_autocorrelation
        rho = rolling_autocorrelation(sample_prices, window=30)
        valid = rho[~np.isnan(rho)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)

    def test_compute_atr_positive(self, sample_prices):
        from production.engines.signal_engine_mean_reversion import compute_atr
        high = sample_prices * 1.005
        low  = sample_prices * 0.995
        atr = compute_atr(high, low, sample_prices, period=14)
        valid = atr[~np.isnan(atr)]
        assert np.all(valid > 0)

    def test_signal_returns_flat_on_short_series(self, sample_bar_array):
        from production.engines.signal_engine_mean_reversion import (
            MeanReversionEngine, SignalDirection
        )
        class FakeConfig:
            def instrument(self, sym):
                return {"mean_reversion_threshold": 2.0, "mean_reversion_window": 30,
                        "atr_period": 20, "time_stop_hours": 3}
            @property
            def bayesian_params(self):
                return {"ic_prior_alpha": 2, "ic_prior_beta": 30, "uncertainty_aversion": 0.5}

        engine = MeanReversionEngine(FakeConfig())

        class ShortBarArray:
            n = 10
            close = np.array([1.0] * 10)
            high  = np.array([1.01] * 10)
            low   = np.array([0.99] * 10)
            def returns(self, log=True): return np.diff(self.close)
            timestamps = [datetime.now(timezone.utc)] * 10

        sig = engine.compute_signal(ShortBarArray(), "EURUSD")
        assert sig.direction == SignalDirection.FLAT
        assert sig.suspended_reason is not None


# ── Trend Following Engine ────────────────────────────────────────────────────
class TestTrendFollowingIndicators:
    def test_ema_length(self, sample_prices):
        from production.engines.signal_engine_trend_following import exponential_moving_average
        ema = exponential_moving_average(sample_prices, period=20)
        assert len(ema) == len(sample_prices)

    def test_ema_nan_prefix(self, sample_prices):
        from production.engines.signal_engine_trend_following import exponential_moving_average
        ema = exponential_moving_average(sample_prices, period=20)
        assert np.all(np.isnan(ema[:19]))
        assert not np.isnan(ema[19])

    def test_donchian_upper_gte_lower(self, sample_prices):
        from production.engines.signal_engine_trend_following import donchian_channel
        high = sample_prices * 1.005
        low  = sample_prices * 0.995
        upper, lower = donchian_channel(high, low, period=20)
        valid = ~(np.isnan(upper) | np.isnan(lower))
        assert np.all(upper[valid] >= lower[valid])

    def test_momentum_12m(self, sample_prices):
        from production.engines.signal_engine_trend_following import momentum_12m
        mom = momentum_12m(sample_prices)
        assert isinstance(mom, float)
        assert not np.isnan(mom)

    def test_momentum_12m_insufficient_data(self):
        from production.engines.signal_engine_trend_following import momentum_12m
        short = np.array([1.0, 1.01, 1.02])
        assert np.isnan(momentum_12m(short))


# ── Volatility Breakout Engine ────────────────────────────────────────────────
class TestVolatilityBreakout:
    def test_atr_ratio_positive(self, sample_prices):
        from production.engines.signal_engine_volatility_breakout import atr_volatility_ratio
        high = sample_prices * 1.005
        low  = sample_prices * 0.995
        vr = atr_volatility_ratio(high, low, sample_prices)
        valid = vr[~np.isnan(vr)]
        assert np.all(valid > 0)

    def test_bollinger_width_positive(self, sample_prices):
        from production.engines.signal_engine_volatility_breakout import bollinger_band_width
        bw = bollinger_band_width(sample_prices, period=20)
        valid = bw[~np.isnan(bw)]
        assert np.all(valid >= 0)


# ── Regime Engine ─────────────────────────────────────────────────────────────
class TestRegimeEngine:
    def test_compute_adx_length(self, sample_prices):
        from production.engines.regime_engine import compute_adx
        high = sample_prices * 1.005
        low  = sample_prices * 0.995
        adx, plus_di, minus_di = compute_adx(high, low, sample_prices, period=14)
        assert len(adx) == len(sample_prices)

    def test_adx_non_negative(self, sample_prices):
        from production.engines.regime_engine import compute_adx
        high = sample_prices * 1.005
        low  = sample_prices * 0.995
        adx, _, _ = compute_adx(high, low, sample_prices, period=14)
        valid = adx[~np.isnan(adx)]
        assert np.all(valid >= 0)

    def test_normalize_strategy_scales_sum_one(self):
        from production.engines.regime_engine import normalize_strategy_scales
        scales = {"trend": 0.80, "mean_rev": 0.10, "breakout": 0.40}
        norm = normalize_strategy_scales(scales)
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_normalize_no_change_if_below_one(self):
        from production.engines.regime_engine import normalize_strategy_scales
        scales = {"trend": 0.30, "mean_rev": 0.20, "breakout": 0.10}
        norm = normalize_strategy_scales(scales)
        for k in scales:
            assert abs(norm[k] - scales[k]) < 1e-9

    def test_cusum_alert_on_declining_pnl(self):
        from production.engines.regime_engine import cusum_structural_break
        # First 60 good, then structural break
        pnl = np.concatenate([np.random.normal(100, 50, 60),
                               np.random.normal(-200, 50, 60)])
        result = cusum_structural_break(pnl)
        assert "alert" in result
        assert "cusum_value" in result


# ── Bayesian Estimator ────────────────────────────────────────────────────────
class TestBayesianEstimator:
    def test_prior_mean(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator(alpha_prior=2, beta_prior=30)
        assert abs(est.posterior_mean - 2/32) < 1e-6

    def test_update_increases_mean_on_wins(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator(alpha_prior=2, beta_prior=30)
        prior_mean = est.posterior_mean
        est.update(n_wins=70, n_total=100)
        assert est.posterior_mean > prior_mean

    def test_update_decreases_mean_on_losses(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator(alpha_prior=2, beta_prior=30)
        prior_mean = est.posterior_mean
        est.update(n_wins=10, n_total=100)
        assert est.posterior_mean < prior_mean

    def test_confidence_weight_bounded(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator()
        est.update(100, 200)
        w = est.confidence_weight(breakeven_ic=0.04)
        assert 0.0 <= w <= 1.0

    def test_n_trades_tracked(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator()
        est.update(30, 50)
        est.update(20, 40)
        assert est.n_trades == 90

    def test_reset_to_prior(self):
        from production.engines.bayesian_estimator import ICBayesianEstimator
        est = ICBayesianEstimator(alpha_prior=2, beta_prior=30)
        est.update(100, 200)
        original_mean = est.posterior_mean
        est.reset()
        assert abs(est.posterior_mean - 2/32) < 1e-6
        assert est.n_trades == 0


# ── Risk Engine ───────────────────────────────────────────────────────────────
class TestRiskEngine:
    def _make_config(self):
        class FakeConfig:
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
                "decay_monitor": {},
                "heartbeat": {"timeout_seconds": 120, "monitor_interval_seconds": 30,
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
            def instrument(self, sym): return {"contract_size": 100000, "lot_size": 0.01}
            def instruments(self): return {}
        return FakeConfig()

    def _make_request(self, balance=100_000):
        from production.engines.risk_engine import PositionSizeRequest
        return PositionSizeRequest(
            instrument="EURUSD", strategy="mean_reversion",
            direction="buy", entry_price=1.1000, stop_loss_price=1.0950,
            signal_confidence=0.8, regime_scale=0.8,
            current_regime="normal_trending",
            account_balance=balance,
            current_portfolio_risk_usd=0.0,
            current_drawdown_pct=0.0,
        )

    def test_position_size_approved(self):
        from production.engines.risk_engine import RiskEngine, RiskCheckResult
        config = self._make_config()
        engine = RiskEngine(config, 100_000)
        req = self._make_request()
        returns = np.random.normal(0, 0.001, 30)
        result = engine.size_position(req, returns, {"contract_size": 100000, "lot_size": 0.01})
        assert result.result in (RiskCheckResult.APPROVED, RiskCheckResult.REDUCED)
        assert result.position_size_lots >= 0.01

    def test_kill_switch_blocks_sizing(self):
        from production.engines.risk_engine import RiskEngine, RiskCheckResult
        config = self._make_config()
        engine = RiskEngine(config, 100_000)
        engine._kill_switch_triggered = True
        req = self._make_request()
        returns = np.random.normal(0, 0.001, 30)
        result = engine.size_position(req, returns, {"contract_size": 100000, "lot_size": 0.01})
        assert result.result == RiskCheckResult.REJECTED

    def test_drawdown_tracker_peak(self):
        from production.engines.risk_engine import DrawdownTracker
        tracker = DrawdownTracker(100_000)
        tracker.update(105_000)
        assert tracker._peak == 105_000
        tracker.update(95_000)
        assert abs(tracker.drawdown_pct - 10_000/105_000) < 1e-6

    def test_kill_switch_reset_requires_name(self):
        from production.engines.risk_engine import RiskEngine
        config = self._make_config()
        engine = RiskEngine(config, 100_000)
        engine._kill_switch_triggered = True
        with pytest.raises(ValueError):
            engine.reset_kill_switch(authorised_by="")
        engine.reset_kill_switch(authorised_by="Test Operator")
        assert not engine.kill_switch_active


# ── Normalize strategy scales ─────────────────────────────────────────────────
class TestNormalizeStrategyScales:
    def test_scales_sum_to_one(self):
        from production.engines.regime_engine import normalize_strategy_scales
        for scales in [
            {"a": 0.8, "b": 0.4, "c": 0.6},
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 0.5, "b": 0.5, "c": 0.5},
        ]:
            norm = normalize_strategy_scales(scales)
            assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_below_one_unchanged(self):
        from production.engines.regime_engine import normalize_strategy_scales
        scales = {"a": 0.3, "b": 0.2, "c": 0.1}
        norm = normalize_strategy_scales(scales)
        for k in scales:
            assert abs(norm[k] - scales[k]) < 1e-9
