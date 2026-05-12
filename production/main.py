"""
main.py — Production Process Orchestrator
Wires all system components together and runs the main trading loop.

Startup sequence:
    1.  Load and validate configuration
    2.  Initialise structured logger
    3.  Connect to MT5
    4.  Prefetch all market data
    5.  Initialise all engines (regime, alpha, risk, portfolio)
    6.  Initialise monitoring (heartbeat, decay, performance)
    7.  Register all components with heartbeat supervisor
    8.  Start heartbeat supervisor thread
    9.  Run initial regime detection
    10. Reconcile open positions with MT5
    11. Enter main loop

Main loop (runs on each new bar):
    a. Update equity / drawdown
    b. Check heartbeat
    c. Refresh market data
    d. Detect regime (daily)
    e. Compute signals for all engines × instruments
    f. Size and submit approved signals
    g. Check exits on open positions
    h. Run daily tasks (end-of-day)
    i. Sleep until next bar

Shutdown:
    - Graceful: close all positions, flush logs, disconnect MT5
    - Emergency: kill switch closes positions immediately

Environment requirement:
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER must be set as environment variables.
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID for alerts (optional but recommended).
"""

import logging
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging setup (before any imports that log) ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            ROOT / "production" / "logs" / "system" / "main.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")

# ── Component imports ─────────────────────────────────────────────────────────
from production.config.config import load_config, SystemConfig
from production.data.feed_manager import FeedManager
from production.data.data_validator import Bar

from production.engines.regime_engine import RegimeEngine, RegimeState
from production.engines.signal_engine_mean_reversion import MeanReversionEngine
from production.engines.signal_engine_trend_following import TrendFollowingEngine
from production.engines.signal_engine_volatility_breakout import VolatilityBreakoutEngine
from production.engines.bayesian_estimator import StrategyEstimatorRegistry
from production.engines.risk_engine import RiskEngine, PositionSizeRequest
from production.engines.portfolio_engine import PortfolioEngine

from production.execution.mt5_bridge import MT5Bridge
from production.execution.order_manager import OrderManager

from production.monitoring.heartbeat import HeartbeatSupervisor, AlertManager
from production.monitoring.decay_monitor import AlphaDecayMonitor
from production.monitoring.performance_monitor import PerformanceMonitor
from production.monitoring.structured_logger import StructuredLogger


# ─────────────────────────────────────────────────────────────────────────────
# TIMEFRAME → BAR INTERVAL SECONDS (for sleep calculation)
# ─────────────────────────────────────────────────────────────────────────────
TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


# ─────────────────────────────────────────────────────────────────────────────
# TRADING SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class TradingSystem:
    """
    Production trading system orchestrator.
    Owns all components and manages their lifecycle.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self._running = False
        self._shutdown_requested = False
        self._last_regime_detect: Optional[datetime] = None
        self._last_daily_tasks: Optional[datetime] = None
        self._current_regime: Optional[RegimeState] = None
        self._prev_regime_key: Optional[tuple] = None

        # ── Initialise structured logger first ────────────────────────────────
        self.slog = StructuredLogger(
            log_base_dir=str(ROOT / "production" / "logs")
        )
        logger.info("Structured logger initialised")

        # ── Alert manager (needs config for Telegram) ─────────────────────────
        self.alerts = AlertManager(config)

        # ── MT5 bridge ────────────────────────────────────────────────────────
        self.mt5 = MT5Bridge(config)

        # ── Data feed ─────────────────────────────────────────────────────────
        self.feed = FeedManager(config)

        # ── Alpha engines ─────────────────────────────────────────────────────
        self.regime_engine = RegimeEngine(config)
        self.mr_engine = MeanReversionEngine(config)
        self.tf_engine = TrendFollowingEngine(config)
        self.vb_engine = VolatilityBreakoutEngine(config)

        # ── Bayesian estimators ───────────────────────────────────────────────
        self.estimators = StrategyEstimatorRegistry(config)

        # ── Risk engine (needs initial balance — set after MT5 connect) ───────
        self.risk_engine: Optional[RiskEngine] = None

        # ── Portfolio engine ──────────────────────────────────────────────────
        self.portfolio_engine = PortfolioEngine(config)

        # ── Order manager (needs risk engine — set after connect) ─────────────
        self.order_manager: Optional[OrderManager] = None

        # ── Monitoring ────────────────────────────────────────────────────────
        self.decay_monitor = AlphaDecayMonitor(config)
        self.perf_monitor: Optional[PerformanceMonitor] = None
        self.heartbeat: Optional[HeartbeatSupervisor] = None

        logger.info(
            f"TradingSystem initialised | "
            f"env={config.deployment.get('environment')} | "
            f"instruments={config.active_instruments}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        """
        Full startup sequence. Returns True if ready to trade.
        Any step failure aborts startup safely.
        """
        logger.info("=" * 60)
        logger.info("TRADING SYSTEM STARTUP")
        logger.info("=" * 60)

        self.slog.log_system("startup_begin", {
            "environment": self.config.deployment.get("environment"),
            "framework_version": self.config.deployment.get("framework_version"),
            "instruments": self.config.active_instruments,
        })

        # ── Step 3: Connect to MT5 ────────────────────────────────────────────
        logger.info("Connecting to MT5...")
        if not self.mt5.connect():
            logger.critical("MT5 connection failed — aborting startup")
            self.alerts.critical("MT5 connection failed at startup")
            return False

        # ── Get initial account balance ───────────────────────────────────────
        account = self.mt5.get_account_info()
        if account is None:
            logger.critical("Cannot read account info — aborting startup")
            return False

        initial_balance = account["balance"]
        logger.info(
            f"Account: balance={initial_balance:,.2f} {account['currency']} | "
            f"equity={account['equity']:,.2f}"
        )

        # ── Initialise balance-dependent components ───────────────────────────
        self.risk_engine = RiskEngine(self.config, initial_balance)
        self.perf_monitor = PerformanceMonitor(initial_balance, self.config)
        self.order_manager = OrderManager(
            self.mt5,
            self.risk_engine,
            self.risk_engine.portfolio_tracker,
            self.alerts,
        )

        # ── Step 4: Prefetch all market data ──────────────────────────────────
        logger.info("Prefetching market data...")
        timeframes = list(set(self.config.timeframes.values()))
        prefetch_results = self.feed.prefetch_all(
            instruments=self.config.active_instruments,
            timeframes=timeframes,
            n_bars=300,
        )
        failed = [k for k, ok in prefetch_results.items() if not ok]
        if failed:
            logger.warning(f"Prefetch failures: {failed}")
            if len(failed) > len(prefetch_results) * 0.3:
                logger.critical("Too many prefetch failures — aborting startup")
                return False

        # ── Step 5: Initial regime detection ─────────────────────────────────
        logger.info("Running initial regime detection...")
        self._detect_regime()

        # ── Step 6: Reconcile open positions ─────────────────────────────────
        logger.info("Reconciling open positions with MT5...")
        recon = self.order_manager.reconcile_with_mt5()
        logger.info(f"Reconciliation: {recon}")

        # ── Step 7: Initialise heartbeat supervisor ───────────────────────────
        self.heartbeat = HeartbeatSupervisor(self.mt5, self.alerts, self.config)
        for component in ["main_loop", "signal_engine", "risk_engine", "data_feed"]:
            self.heartbeat.register(component)
        self.heartbeat.start()

        # ── Register signal handlers for graceful shutdown ────────────────────
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        logger.info("=" * 60)
        logger.info("STARTUP COMPLETE — entering main loop")
        logger.info("=" * 60)
        self.slog.log_system("startup_complete", {"initial_balance": initial_balance})
        self.alerts.info(
            f"System started | balance={initial_balance:,.2f} | "
            f"regime={self._current_regime.log_summary() if self._current_regime else 'unknown'}"
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> None:
        """
        Main trading loop. Runs until shutdown is requested.
        Sleeps between iterations based on shortest active timeframe.
        """
        primary_tf = self.config.timeframes.get("mean_reversion", "M5")
        bar_interval = TIMEFRAME_SECONDS.get(primary_tf, 300)

        logger.info(f"Main loop running on {primary_tf} bars ({bar_interval}s interval)")

        while self._running and not self._shutdown_requested:
            loop_start = time.time()

            try:
                self._main_loop_iteration()
            except Exception as e:
                logger.exception(f"Main loop iteration error: {e}")
                self.slog.log_system("loop_error", {"error": str(e)}, level="error")
                self.alerts.warning(f"Main loop error: {e}")
                # Don't crash — log and continue on next bar

            # ── Sleep until next bar ──────────────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_time = max(1.0, bar_interval - elapsed)
            logger.debug(f"Loop elapsed={elapsed:.1f}s, sleeping {sleep_time:.1f}s")
            self._interruptible_sleep(sleep_time)

        logger.info("Main loop exited")

    def _main_loop_iteration(self) -> None:
        """Single iteration of the main trading loop."""
        now = datetime.now(timezone.utc)

        # ── (a) Heartbeat ─────────────────────────────────────────────────────
        if self.heartbeat:
            self.heartbeat.beat("main_loop")

        # ── (b) Update equity ─────────────────────────────────────────────────
        account = self.mt5.get_account_info()
        if account:
            equity = account["equity"]
            self.risk_engine.update_equity(equity)
            self.perf_monitor.update_equity(equity)
            self._log_drawdown_if_needed(equity)

        # ── (c) Refresh market data ───────────────────────────────────────────
        self.heartbeat.beat("data_feed")
        bar_arrays = self._refresh_data()
        if not bar_arrays:
            logger.warning("No valid bar data — skipping this iteration")
            return

        # ── (d) Regime detection (once per day) ───────────────────────────────
        if self._should_detect_regime(now):
            self._detect_regime(bar_arrays)

        if self._current_regime is None:
            logger.warning("No regime detected yet — skipping signal generation")
            return

        # ── Kill switch check ─────────────────────────────────────────────────
        if self.heartbeat and self.heartbeat.kill_switch_fired:
            logger.warning("Kill switch active — no new positions")
            return
        if self.risk_engine.kill_switch_active:
            logger.warning("Risk engine kill switch active — no new positions")
            return

        # ── (e) Compute signals and size/submit ───────────────────────────────
        self.heartbeat.beat("signal_engine")
        self.heartbeat.beat("risk_engine")
        self._process_signals(bar_arrays, now)

        # ── (f) Check exits ───────────────────────────────────────────────────
        mr_bars = {
            sym: bar_arrays.get(f"{sym}_{self.config.timeframes['mean_reversion']}")
            for sym in self.config.active_instruments
        }
        mr_bars_clean = {k: v for k, v in mr_bars.items() if v is not None}
        closed = self.order_manager.check_exits(
            bar_arrays=mr_bars_clean,
            mean_rev_engine=self.mr_engine,
            breakout_engine=self.vb_engine,
        )
        if closed:
            logger.info(f"Exit checks closed {len(closed)} positions: {closed}")

        # ── (g) Daily tasks ───────────────────────────────────────────────────
        if self._should_run_daily_tasks(now):
            self._run_daily_tasks(now, account)

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    def _process_signals(
        self,
        bar_arrays: dict,
        now: datetime,
    ) -> None:
        """
        Compute signals for all active engines × instruments.
        Sizes and submits approved signals.
        """
        regime = self._current_regime
        scales = self.regime_engine.get_strategy_scales(regime)

        for instrument in self.config.active_instruments:
            instr_cfg = self.config.instrument(instrument)
            account = self.mt5.get_account_info()
            if account is None:
                continue
            balance = account["balance"]

            # ── Mean Reversion ────────────────────────────────────────────────
            if self.config.active_strategies.get("mean_reversion", True):
                mr_key = f"{instrument}_{self.config.timeframes['mean_reversion']}"
                mr_bars = bar_arrays.get(mr_key)
                if mr_bars and mr_bars.n >= 50:
                    self._process_mean_reversion(
                        instrument, mr_bars, instr_cfg,
                        scales.mean_reversion, balance, regime
                    )

            # ── Trend Following ───────────────────────────────────────────────
            if self.config.active_strategies.get("trend_following", True):
                tf_key = f"{instrument}_{self.config.timeframes['trend_following']}"
                tf_bars = bar_arrays.get(tf_key)
                if tf_bars and tf_bars.n >= 120:
                    self._process_trend_following(
                        instrument, tf_bars, instr_cfg,
                        scales.trend_following, balance, regime
                    )

            # ── Volatility Breakout ───────────────────────────────────────────
            if self.config.active_strategies.get("volatility_breakout", True):
                vb_key = f"{instrument}_{self.config.timeframes['volatility_breakout']}"
                vb_bars = bar_arrays.get(vb_key)
                if vb_bars and vb_bars.n >= 80:
                    self._process_volatility_breakout(
                        instrument, vb_bars, instr_cfg,
                        scales.volatility_breakout, balance, regime
                    )

    def _process_mean_reversion(
        self, instrument, bar_array, instr_cfg,
        regime_scale, balance, regime
    ) -> None:
        signal = self.mr_engine.compute_signal(bar_array, instrument, regime_scale)

        self.slog.log_signal(
            strategy="mean_reversion", symbol=instrument,
            direction=signal.direction.name,
            signal_strength=signal.signal_strength,
            z_score=signal.z_score_adj,
            regime=str(regime.regime_key),
            regime_scale=regime_scale,
            ic_posterior_mean=self.estimators.get_ic_estimator(
                "mean_reversion", instrument
            ).posterior_mean,
            actionable=signal.is_actionable,
            suspended_reason=signal.suspended_reason,
        )

        if not signal.is_actionable:
            return

        self._submit_signal(
            strategy="mean_reversion",
            instrument=instrument,
            direction=signal.direction.name.lower(),
            entry_price=signal.entry_price,
            stop_distance=signal.stop_distance_pips,
            signal_confidence=self.estimators.get_confidence_weight(
                "mean_reversion", instrument
            ),
            regime_scale=regime_scale,
            balance=balance,
            regime=regime,
            instr_cfg=instr_cfg,
            bar_array=bar_array,
            entry_z_score=signal.z_score_adj,
        )

    def _process_trend_following(
        self, instrument, bar_array, instr_cfg,
        regime_scale, balance, regime
    ) -> None:
        signal = self.tf_engine.compute_signal(
            bar_array, instrument,
            timeframe=self.config.timeframes["trend_following"],
            regime_scale=regime_scale,
            is_daily=True,
        )

        self.slog.log_signal(
            strategy="trend_following", symbol=instrument,
            direction=signal.direction.name,
            signal_strength=signal.signal_strength,
            trend_strength=signal.trend_strength,
            regime=str(regime.regime_key),
            regime_scale=regime_scale,
            ic_posterior_mean=self.estimators.get_ic_estimator(
                "trend_following", instrument
            ).posterior_mean,
            actionable=signal.is_actionable,
            suspended_reason=signal.suspended_reason,
        )

        if not signal.is_actionable:
            return

        stop_distance = signal.atr * 2.0  # 2× ATR for trend following
        self._submit_signal(
            strategy="trend_following",
            instrument=instrument,
            direction=signal.direction.name.lower(),
            entry_price=signal.entry_price,
            stop_distance=stop_distance,
            signal_confidence=(
                self.estimators.get_confidence_weight("trend_following", instrument)
                * signal.allocation_scale
            ),
            regime_scale=regime_scale,
            balance=balance,
            regime=regime,
            instr_cfg=instr_cfg,
            bar_array=bar_array,
        )

    def _process_volatility_breakout(
        self, instrument, bar_array, instr_cfg,
        regime_scale, balance, regime
    ) -> None:
        signal = self.vb_engine.compute_signal(bar_array, instrument, regime_scale)

        self.slog.log_signal(
            strategy="volatility_breakout", symbol=instrument,
            direction=signal.direction.name,
            signal_strength=signal.signal_strength,
            regime=str(regime.regime_key),
            regime_scale=regime_scale,
            ic_posterior_mean=self.estimators.get_ic_estimator(
                "volatility_breakout", instrument
            ).posterior_mean,
            actionable=signal.is_actionable,
            suspended_reason=signal.suspended_reason,
        )

        if not signal.is_actionable:
            return

        stop_distance = signal.atr * self.config.instruments.get(
            "instruments", {}
        ).get(instrument, {}).get("atr_period", 1.0)

        self._submit_signal(
            strategy="volatility_breakout",
            instrument=instrument,
            direction=signal.direction.name.lower(),
            entry_price=signal.entry_price,
            stop_distance=stop_distance if stop_distance > 0 else signal.atr,
            signal_confidence=self.estimators.get_confidence_weight(
                "volatility_breakout", instrument
            ),
            regime_scale=regime_scale,
            balance=balance,
            regime=regime,
            instr_cfg=instr_cfg,
            bar_array=bar_array,
            take_profit=signal.profit_target,
        )

    def _submit_signal(
        self,
        strategy: str,
        instrument: str,
        direction: str,
        entry_price: float,
        stop_distance: float,
        signal_confidence: float,
        regime_scale: float,
        balance: float,
        regime: RegimeState,
        instr_cfg: dict,
        bar_array,
        entry_z_score: Optional[float] = None,
        take_profit: float = 0.0,
    ) -> None:
        """Build size request, run risk checks, submit order if approved."""
        if stop_distance <= 0:
            logger.warning(f"Invalid stop_distance={stop_distance} for {instrument} — skipping")
            return

        stop_price = (
            entry_price - stop_distance if direction == "buy"
            else entry_price + stop_distance
        )

        regime_key = "_".join(regime.regime_key)

        request = PositionSizeRequest(
            instrument=instrument,
            strategy=strategy,
            direction=direction,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            signal_confidence=signal_confidence,
            regime_scale=regime_scale,
            current_regime=regime_key,
            account_balance=balance,
            current_portfolio_risk_usd=self.risk_engine.portfolio_tracker.total_open_risk_usd,
            current_drawdown_pct=self.risk_engine.drawdown_tracker.drawdown_pct,
        )

        size_result = self.risk_engine.size_position(
            request=request,
            instrument_returns=bar_array.returns(),
            instrument_config=instr_cfg,
        )

        if not size_result.is_approved:
            logger.debug(
                f"Signal rejected: {instrument} {strategy} — "
                f"{size_result.rejection_reason}"
            )
            return

        # Spread check
        tick = self.mt5.get_latest_tick(instrument)
        if tick:
            spread_ok, spread_reason = self.risk_engine.check_spread(
                instrument=instrument,
                current_spread_points=(tick["ask"] - tick["bid"]) /
                    instr_cfg.get("point_value", 0.0001),
                instrument_config=instr_cfg,
            )
            if not spread_ok:
                logger.info(f"Signal blocked by spread: {instrument} — {spread_reason}")
                return

        # Submit
        ticket = self.order_manager.open_position(
            size_result=size_result,
            instrument_config=instr_cfg,
            take_profit=take_profit,
            entry_z_score=entry_z_score,
            comment=f"{strategy[:4]}_{regime_key[:6]}",
        )

        if ticket:
            self.slog.log_trade_open(
                ticket=ticket,
                symbol=instrument,
                strategy=strategy,
                direction=direction,
                volume=size_result.position_size_lots,
                entry_price=entry_price,
                stop_loss=stop_price,
                take_profit=take_profit,
                dollar_risk=size_result.dollar_risk,
                signal_confidence=signal_confidence,
                regime=regime_key,
            )
            self.risk_engine.portfolio_tracker.add_position(
                ticket=ticket,
                instrument=instrument,
                strategy=strategy,
                direction=direction,
                size_lots=size_result.position_size_lots,
                entry_price=entry_price,
                stop_price=stop_price,
                dollar_risk=size_result.dollar_risk,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_regime(self, bar_arrays: Optional[dict] = None) -> None:
        """Run regime detection and log if regime changed."""
        try:
            if bar_arrays is None:
                bar_arrays = self._refresh_data()
            if not bar_arrays:
                return

            # Use SPX500 daily as primary regime instrument
            primary = bar_arrays.get(
                f"SPX500_{self.config.timeframes['trend_following']}"
            )
            gold = bar_arrays.get(
                f"XAUUSD_{self.config.timeframes['trend_following']}"
            )
            fx = bar_arrays.get(
                f"EURUSD_{self.config.timeframes['mean_reversion']}"
            )

            if primary is None:
                primary = next(
                    (v for k, v in bar_arrays.items() if "D1" in k or "SPX" in k),
                    None
                )
            if primary is None:
                logger.warning("No primary bar array for regime detection")
                return

            new_regime = self.regime_engine.compute_regime(
                equity_index_bar_array=primary,
                fx_bar_array=fx,
                gold_bar_array=gold,
            )

            # Log regime change
            if self._prev_regime_key != new_regime.regime_key:
                old_key = self._prev_regime_key or ("unknown", "unknown")
                self.slog.log_regime_change(
                    old_regime="_".join(old_key),
                    new_regime="_".join(new_regime.regime_key),
                    adx=new_regime.adx,
                    vol_percentile=new_regime.vol_percentile,
                    crisis_active=new_regime.crisis_active,
                    strategy_scales=self.regime_engine.get_strategy_scales(
                        new_regime
                    ).as_dict(),
                )
                if self._prev_regime_key is not None:
                    self.alerts.warning(
                        f"Regime change: {old_key} → {new_regime.regime_key}"
                    )
                self._prev_regime_key = new_regime.regime_key

            self._current_regime = new_regime
            self._last_regime_detect = datetime.now(timezone.utc)

        except Exception as e:
            logger.exception(f"Regime detection error: {e}")

    def _should_detect_regime(self, now: datetime) -> bool:
        if self._last_regime_detect is None:
            return True
        return (now - self._last_regime_detect).total_seconds() >= 86400  # Daily

    # ─────────────────────────────────────────────────────────────────────────
    # DATA REFRESH
    # ─────────────────────────────────────────────────────────────────────────
    def _refresh_data(self) -> dict:
        """
        Refresh market data for all active instruments and timeframes.
        Returns {symbol_tf: BarArray} dict.
        """
        bar_arrays = {}
        timeframes = list(set(self.config.timeframes.values()))

        for instrument in self.config.active_instruments:
            for tf in timeframes:
                key = f"{instrument}_{tf}"
                bars = self.feed.get_or_fetch(
                    instrument=instrument,
                    timeframe=tf,
                    n_bars=300,
                    max_cache_age_seconds=TIMEFRAME_SECONDS.get(tf, 300),
                )
                if bars is not None:
                    bar_arrays[key] = bars

        return bar_arrays

    # ─────────────────────────────────────────────────────────────────────────
    # DAILY TASKS
    # ─────────────────────────────────────────────────────────────────────────
    def _run_daily_tasks(self, now: datetime, account: Optional[dict]) -> None:
        """
        End-of-day tasks:
        - Run decay monitor for all strategies
        - Export performance summary
        - Portfolio rebalance check
        - Log daily summary
        """
        logger.info("Running daily tasks...")

        # ── Decay monitoring ──────────────────────────────────────────────────
        strategy_instrument_pairs = [
            (strategy, instrument)
            for strategy in ["mean_reversion", "trend_following", "volatility_breakout"]
            for instrument in self.config.active_instruments
        ]
        decay_results = self.decay_monitor.compute_all(strategy_instrument_pairs)

        for key, metrics in decay_results.items():
            if metrics.n_conditions > 0:
                self.slog.log_decay_alert(
                    strategy=metrics.strategy,
                    instrument=metrics.instrument,
                    response=metrics.response.value,
                    n_conditions=metrics.n_conditions,
                    active_reasons=metrics.active_reasons,
                    metrics={
                        "rolling_sharpe_30d": metrics.rolling_sharpe_30d,
                        "rolling_ic_30d": metrics.rolling_ic_30d,
                        "slippage_ratio": metrics.slippage_ratio,
                        "cusum_value": metrics.cusum_value,
                    },
                )

        # ── Performance snapshot ──────────────────────────────────────────────
        if account and self.perf_monitor:
            self.risk_engine.record_daily_pnl(
                account["equity"] - account["balance"]
            )
            summary = self.perf_monitor.export_daily_summary(
                current_equity=account["equity"],
                current_balance=account["balance"],
            )
            self.slog.log_performance(summary)
            logger.info(
                f"Daily summary: equity={account['equity']:,.2f} | "
                f"trades={summary.get('n_trades_today', 0)}"
            )

        # ── Heartbeat log ─────────────────────────────────────────────────────
        if self.heartbeat and account:
            self.slog.log_heartbeat(
                component_statuses=self.heartbeat.component_statuses(),
                mt5_connected=self.mt5.is_connected(),
                equity=account.get("equity", 0),
                n_open_positions=len(self.order_manager.open_positions),
            )

        self._last_daily_tasks = now
        logger.info("Daily tasks complete")

    def _should_run_daily_tasks(self, now: datetime) -> bool:
        """Run daily tasks once per day around 22:30 UTC (after NY close)."""
        if self._last_daily_tasks is None:
            return False  # Don't run on first iteration
        if (now - self._last_daily_tasks).total_seconds() < 86400 * 0.9:
            return False
        # Prefer to run after 22:00 UTC
        return now.hour >= 22

    # ─────────────────────────────────────────────────────────────────────────
    # MONITORING HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _log_drawdown_if_needed(self, equity: float) -> None:
        """Log drawdown state when it changes materially."""
        level = self.risk_engine.drawdown_tracker.drawdown_level
        dd = self.risk_engine.drawdown_tracker.drawdown_pct
        if dd > 0.03:
            self.slog.log_drawdown(
                drawdown_pct=dd,
                drawdown_level=level.value,
                daily_loss_pct=self.risk_engine.drawdown_tracker.daily_loss_pct,
                equity=equity,
                peak_equity=self.risk_engine.drawdown_tracker._peak,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────
    def _signal_handler(self, signum, frame) -> None:
        """Handle SIGINT/SIGTERM — request graceful shutdown."""
        logger.info(f"Signal {signum} received — requesting graceful shutdown")
        self._shutdown_requested = True
        self._running = False

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in small increments so shutdown signals are caught quickly."""
        end = time.time() + seconds
        while time.time() < end and not self._shutdown_requested:
            time.sleep(min(1.0, end - time.time()))

    def shutdown(self, graceful: bool = True) -> None:
        """
        Shutdown the trading system.
        graceful=True: close positions cleanly before disconnecting.
        graceful=False: kill switch first.
        """
        self._running = False
        logger.info(f"Shutdown initiated (graceful={graceful})")

        if not graceful and self.heartbeat:
            self.heartbeat.trigger_kill_switch("manual_shutdown")

        # Stop heartbeat supervisor
        if self.heartbeat:
            self.heartbeat.stop()

        # Graceful position close
        if graceful and self.order_manager:
            open_pos = self.order_manager.open_positions
            if open_pos:
                logger.info(f"Closing {len(open_pos)} open positions...")
                for ticket in list(open_pos.keys()):
                    self.order_manager.close_position(ticket, "graceful_shutdown")

        # Flush logs
        self.slog.log_system("shutdown", {"graceful": graceful})
        self.slog.flush_all()
        self.slog.close()

        # Disconnect MT5
        self.mt5.disconnect()
        logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """
    Main entry point. Returns exit code (0=success, 1=error).
    """
    logger.info("Loading configuration...")
    try:
        config = load_config()
    except Exception as e:
        logger.critical(f"Configuration load failed: {e}")
        return 1

    # Safety check — never start live without explicit environment flag
    if config.is_live():
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING MODE ACTIVE")
        logger.warning(f"Deployed by: {config.deployment.get('deployed_by')}")
        logger.warning(f"Deployed at: {config.deployment.get('deployed_at')}")
        logger.warning("=" * 60)

    system = TradingSystem(config)

    try:
        if not system.start():
            logger.critical("Startup failed — exiting")
            return 1

        system.run()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received")
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        system.shutdown(graceful=False)
        return 1
    finally:
        system.shutdown(graceful=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
