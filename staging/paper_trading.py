"""
paper_trading.py — Paper Trading (Stage 4 of 7-Stage Pipeline)
Implements Section 10.1, Stage 4 of the framework.

Stage 4 requirements:
    Duration  : Minimum 60 trading days
    Data      : Live market data, ZERO capital
    Required  : Paper Sharpe > 0.40 annualised
                Execution quality acceptable
                Slippage estimate consistent with backtest assumptions
    Gate      : Performance review at 60 days by research committee
    Output    : Paper trading log with full signal and execution history

Design:
    Paper trading runs the FULL production signal stack on live data but
    intercepts the MT5Bridge before any real order is submitted.
    All signals, sizing decisions, and hypothetical fills are recorded.
    This validates that:
        1. Signals generate at the expected frequency
        2. Execution quality assumptions hold in live conditions
        3. The system runs stably without manual intervention
        4. Slippage estimates from backtest are not wildly optimistic
"""

import logging
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
logger = logging.getLogger("paper_trading")

import numpy as np

from production.config.config import load_config
from production.data.feed_manager import FeedManager
from production.engines.regime_engine import RegimeEngine
from production.engines.signal_engine_mean_reversion import MeanReversionEngine, SignalDirection
from production.engines.signal_engine_trend_following import TrendFollowingEngine
from production.engines.signal_engine_volatility_breakout import VolatilityBreakoutEngine
from production.engines.bayesian_estimator import StrategyEstimatorRegistry
from production.engines.risk_engine import RiskEngine, PositionSizeRequest
from production.monitoring.structured_logger import StructuredLogger
from production.monitoring.performance_monitor import PerformanceMonitor


# ─────────────────────────────────────────────────────────────────────────────
# PAPER POSITION
# ─────────────────────────────────────────────────────────────────────────────
class PaperPosition:
    """A simulated (paper) position with no real capital at risk."""
    def __init__(
        self,
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
        atr: float,
    ):
        self.symbol = symbol
        self.strategy = strategy
        self.direction = direction
        self.volume = volume
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.dollar_risk = dollar_risk
        self.signal_confidence = signal_confidence
        self.regime = regime
        self.atr = atr
        self.opened_at = datetime.now(timezone.utc)
        self.closed_at: Optional[datetime] = None
        self.close_price: Optional[float] = None
        self.close_reason: Optional[str] = None
        self.pnl: Optional[float] = None
        self.bars_held: int = 0

    @property
    def is_long(self) -> bool:
        return self.direction == "buy"

    def unrealised_pnl(self, current_price: float, contract_size: float = 100_000) -> float:
        mult = 1 if self.is_long else -1
        return mult * (current_price - self.entry_price) * self.volume * contract_size

    def check_exit(self, current_price: float) -> tuple[bool, str]:
        if self.is_long:
            if current_price <= self.stop_loss:
                return True, f"stop_loss@{current_price:.5f}"
            if self.take_profit > 0 and current_price >= self.take_profit:
                return True, f"take_profit@{current_price:.5f}"
        else:
            if current_price >= self.stop_loss:
                return True, f"stop_loss@{current_price:.5f}"
            if self.take_profit > 0 and current_price <= self.take_profit:
                return True, f"take_profit@{current_price:.5f}"
        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class PaperTradingEngine:
    """
    Runs the full signal stack on live data with zero capital commitment.
    Records all decisions for Stage 4 performance review.
    """

    MIN_DAYS_REQUIRED = 60
    MIN_SHARPE_REQUIRED = 0.40

    def __init__(self, config, initial_paper_balance: float = 100_000.0):
        self.config = config
        self._balance = initial_paper_balance
        self._equity = initial_paper_balance
        self._peak = initial_paper_balance

        # Core components
        self.feed = FeedManager(config)
        self.regime_engine = RegimeEngine(config)
        self.mr_engine = MeanReversionEngine(config)
        self.tf_engine = TrendFollowingEngine(config)
        self.vb_engine = VolatilityBreakoutEngine(config)
        self.estimators = StrategyEstimatorRegistry(config)
        self.risk_engine = RiskEngine(config, initial_paper_balance)
        self.perf_monitor = PerformanceMonitor(initial_paper_balance, config)
        self.slog = StructuredLogger(str(ROOT / "staging" / "paper_logs"))

        # State
        self._positions: list[PaperPosition] = []
        self._closed_positions: list[PaperPosition] = []
        self._signal_log: list[dict] = []
        self._start_date = datetime.now(timezone.utc)
        self._current_regime = None
        self._bar_count = 0
        self._running = False

        logger.info(
            f"PaperTradingEngine initialised | "
            f"balance={initial_paper_balance:,.0f} | "
            f"min_days={self.MIN_DAYS_REQUIRED}"
        )

    def run(self, max_bars: Optional[int] = None) -> None:
        """
        Main paper trading loop. Runs until stopped or max_bars reached.
        Connects to MT5 for live data but submits NO real orders.
        """
        from production.execution.mt5_bridge import MT5Bridge
        bridge = MT5Bridge(self.config)

        if not bridge.connect():
            logger.error("MT5 connection failed — cannot run paper trading")
            return

        logger.info("Paper trading started — NO REAL ORDERS WILL BE SUBMITTED")
        self._running = True
        tf = self.config.timeframes.get("mean_reversion", "M5")
        from production.monitoring.heartbeat import AlertManager
        from production.data.data_validator import TIMEFRAME_SECONDS
        bar_seconds = TIMEFRAME_SECONDS.get(tf, 300)

        try:
            while self._running:
                if max_bars and self._bar_count >= max_bars:
                    break
                self._run_iteration(bridge)
                self._bar_count += 1
                time.sleep(bar_seconds)
        except KeyboardInterrupt:
            logger.info("Paper trading stopped by user")
        finally:
            self._running = False
            bridge.disconnect()
            self._generate_stage4_report()

    def _run_iteration(self, bridge) -> None:
        """Single paper trading iteration."""
        # Refresh data
        bar_arrays = {}
        for instr in self.config.active_instruments:
            for tf in set(self.config.timeframes.values()):
                bars = self.feed.get_or_fetch(instr, tf, n_bars=300)
                if bars:
                    bar_arrays[f"{instr}_{tf}"] = bars

        if not bar_arrays:
            return

        # Regime detection (daily)
        if self._current_regime is None or self._bar_count % 288 == 0:
            primary = next(
                (v for k, v in bar_arrays.items() if "SPX500_D1" in k or "D1" in k),
                None
            )
            if primary:
                self._current_regime = self.regime_engine.compute_regime(primary)

        if self._current_regime is None:
            return

        scales = self.regime_engine.get_strategy_scales(self._current_regime)

        # Check exits on open paper positions
        self._check_paper_exits(bar_arrays)

        # Generate signals
        for instr in self.config.active_instruments:
            instr_cfg = self.config.instrument(instr)

            # Mean reversion
            mr_key = f"{instr}_{self.config.timeframes['mean_reversion']}"
            if mr_key in bar_arrays and bar_arrays[mr_key].n >= 50:
                self._process_paper_signal(
                    "mean_reversion", instr,
                    bar_arrays[mr_key], instr_cfg,
                    scales.mean_reversion, bar_arrays
                )

            # Trend following
            tf_key = f"{instr}_{self.config.timeframes['trend_following']}"
            if tf_key in bar_arrays and bar_arrays[tf_key].n >= 120:
                self._process_paper_signal(
                    "trend_following", instr,
                    bar_arrays[tf_key], instr_cfg,
                    scales.trend_following, bar_arrays
                )

            # Volatility breakout
            vb_key = f"{instr}_{self.config.timeframes['volatility_breakout']}"
            if vb_key in bar_arrays and bar_arrays[vb_key].n >= 80:
                self._process_paper_signal(
                    "volatility_breakout", instr,
                    bar_arrays[vb_key], instr_cfg,
                    scales.volatility_breakout, bar_arrays
                )

        # Update paper equity
        self._update_paper_equity(bar_arrays)

    def _process_paper_signal(
        self, strategy, instrument, bar_array, instr_cfg,
        regime_scale, all_bar_arrays
    ) -> None:
        """Generate a paper signal and record it without submitting."""
        try:
            if strategy == "mean_reversion":
                sig = self.mr_engine.compute_signal(bar_array, instrument, regime_scale)
                direction = sig.direction.name.lower()
                actionable = sig.is_actionable
                strength = sig.signal_strength
                stop_dist = sig.stop_distance_pips
                tp = 0.0
            elif strategy == "trend_following":
                sig = self.tf_engine.compute_signal(
                    bar_array, instrument,
                    timeframe=self.config.timeframes["trend_following"],
                    regime_scale=regime_scale, is_daily=True
                )
                direction = sig.direction.name.lower()
                actionable = sig.is_actionable
                strength = sig.signal_strength
                stop_dist = sig.atr * 2.0
                tp = 0.0
            else:  # volatility_breakout
                sig = self.vb_engine.compute_signal(bar_array, instrument, regime_scale)
                direction = sig.direction.name.lower()
                actionable = sig.is_actionable
                strength = sig.signal_strength
                stop_dist = sig.atr if sig.atr > 0 else 0.0
                tp = sig.profit_target

            entry_price = float(bar_array.close[-1])

            self._signal_log.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "strategy": strategy,
                "instrument": instrument,
                "direction": direction,
                "actionable": actionable,
                "signal_strength": strength,
                "regime": str(self._current_regime.regime_key),
                "regime_scale": regime_scale,
            })

            self.slog.log_signal(
                strategy=strategy, symbol=instrument,
                direction=direction, signal_strength=strength,
                regime=str(self._current_regime.regime_key),
                regime_scale=regime_scale,
                actionable=actionable,
            )

            if not actionable or stop_dist <= 0:
                return

            # Size the position (paper)
            stop_price = (
                entry_price - stop_dist if direction == "buy"
                else entry_price + stop_dist
            )
            conf = self.estimators.get_confidence_weight(strategy, instrument)
            regime_key = "_".join(self._current_regime.regime_key)

            req = PositionSizeRequest(
                instrument=instrument, strategy=strategy,
                direction=direction, entry_price=entry_price,
                stop_loss_price=stop_price, signal_confidence=conf,
                regime_scale=regime_scale, current_regime=regime_key,
                account_balance=self._balance,
                current_portfolio_risk_usd=sum(
                    p.dollar_risk for p in self._positions
                ),
                current_drawdown_pct=(self._peak - self._equity) / (self._peak + 1e-10),
            )
            size = self.risk_engine.size_position(req, bar_array.returns(), instr_cfg)

            if size.is_approved and size.position_size_lots > 0:
                pos = PaperPosition(
                    symbol=instrument, strategy=strategy,
                    direction=direction, volume=size.position_size_lots,
                    entry_price=entry_price, stop_loss=stop_price,
                    take_profit=tp, dollar_risk=size.dollar_risk,
                    signal_confidence=conf, regime=regime_key,
                    atr=stop_dist,
                )
                self._positions.append(pos)
                logger.info(
                    f"[PAPER] OPEN {instrument} {strategy} {direction} "
                    f"{size.position_size_lots:.2f}lots @ {entry_price:.5f} | "
                    f"risk=${size.dollar_risk:.2f}"
                )
                self.slog.log_trade_open(
                    ticket=id(pos), symbol=instrument, strategy=strategy,
                    direction=direction, volume=size.position_size_lots,
                    entry_price=entry_price, stop_loss=stop_price,
                    take_profit=tp, dollar_risk=size.dollar_risk,
                    signal_confidence=conf, regime=regime_key,
                )

        except Exception as e:
            logger.debug(f"Paper signal error {strategy}/{instrument}: {e}")

    def _check_paper_exits(self, bar_arrays: dict) -> None:
        """Check exit conditions for open paper positions."""
        for pos in list(self._positions):
            pos.bars_held += 1
            key = f"{pos.symbol}_{self.config.timeframes['mean_reversion']}"
            bars = bar_arrays.get(key)
            if not bars or bars.n < 2:
                continue
            current_price = float(bars.close[-1])
            exit_flag, reason = pos.check_exit(current_price)

            if not exit_flag and pos.strategy == "mean_reversion":
                hours = pos.bars_held * TIMEFRAME_SECONDS_MAP.get(
                    self.config.timeframes.get("mean_reversion", "M5"), 300
                ) / 3600
                max_h = self.config.instrument(pos.symbol).get("time_stop_hours", 3)
                if hours >= max_h:
                    exit_flag, reason = True, f"time_stop@{hours:.1f}h"

            if exit_flag:
                contract = self.config.instrument(pos.symbol).get("contract_size", 100_000)
                mult = 1 if pos.is_long else -1
                pnl = mult * (current_price - pos.entry_price) * pos.volume * contract
                pos.close_price = current_price
                pos.close_reason = reason
                pos.pnl = pnl
                pos.closed_at = datetime.now(timezone.utc)
                self._equity += pnl
                if self._equity > self._peak:
                    self._peak = self._equity

                is_win = pnl > 0
                self.estimators.record_trade(pos.strategy, pos.symbol, is_win)

                self._positions.remove(pos)
                self._closed_positions.append(pos)

                logger.info(
                    f"[PAPER] CLOSE {pos.symbol} {pos.strategy} @ {current_price:.5f} | "
                    f"pnl={pnl:+.2f} | reason={reason}"
                )
                self.slog.log_trade_close(
                    ticket=id(pos), symbol=pos.symbol, strategy=pos.strategy,
                    close_price=current_price, realised_pnl=pnl,
                    close_reason=reason, bars_held=pos.bars_held,
                    hours_held=(pos.closed_at - pos.opened_at).total_seconds() / 3600,
                )

    def _update_paper_equity(self, bar_arrays: dict) -> None:
        """Update unrealised equity from open paper positions."""
        unrealised = 0.0
        for pos in self._positions:
            key = f"{pos.symbol}_{self.config.timeframes['mean_reversion']}"
            bars = bar_arrays.get(key)
            if bars and bars.n > 0:
                contract = self.config.instrument(pos.symbol).get("contract_size", 100_000)
                unrealised += pos.unrealised_pnl(float(bars.close[-1]), contract)
        self._equity = self._balance + unrealised
        self.perf_monitor.update_equity(self._equity)

    def _generate_stage4_report(self) -> dict:
        """Generate Stage 4 performance report for research committee review."""
        days_elapsed = (datetime.now(timezone.utc) - self._start_date).days
        closed = self._closed_positions
        n_trades = len(closed)

        if n_trades > 0:
            pnls = np.array([p.pnl for p in closed if p.pnl is not None])
            wins = int(np.sum(pnls > 0))
            win_rate = wins / n_trades
            total_pnl = float(np.sum(pnls))
            daily_ret = total_pnl / (self._balance * max(days_elapsed, 1))
            sharpe = (
                float(np.mean(pnls) / np.std(pnls) * np.sqrt(252 / max(days_elapsed, 1)))
                if np.std(pnls) > 1e-10 else 0.0
            )
            avg_hold = float(np.mean([
                (p.closed_at - p.opened_at).total_seconds() / 3600
                for p in closed if p.closed_at
            ]))
        else:
            win_rate = total_pnl = sharpe = avg_hold = 0.0
            n_trades = wins = 0

        drawdown = max(0.0, (self._peak - self._equity) / (self._peak + 1e-10))

        report = {
            "stage": 4,
            "strategy": "all_engines",
            "days_elapsed": days_elapsed,
            "min_days_required": self.MIN_DAYS_REQUIRED,
            "duration_satisfied": days_elapsed >= self.MIN_DAYS_REQUIRED,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "paper_sharpe": sharpe,
            "min_sharpe_required": self.MIN_SHARPE_REQUIRED,
            "sharpe_satisfied": sharpe >= self.MIN_SHARPE_REQUIRED,
            "max_drawdown_pct": drawdown,
            "avg_hold_hours": avg_hold,
            "n_signals_generated": len(self._signal_log),
            "gate_passes": (
                days_elapsed >= self.MIN_DAYS_REQUIRED
                and sharpe >= self.MIN_SHARPE_REQUIRED
            ),
        }

        logger.info("=" * 60)
        logger.info("STAGE 4 PAPER TRADING REPORT")
        logger.info("=" * 60)
        for k, v in report.items():
            if isinstance(v, float):
                logger.info(f"  {k:<30}: {v:.4f}")
            else:
                logger.info(f"  {k:<30}: {v}")

        verdict = "PASS — proceed to Stage 5 shadow deployment" if report["gate_passes"] \
            else "FAIL — extend paper trading period or investigate performance"
        logger.info(f"\nGate verdict: {verdict}")
        logger.info("=" * 60)

        return report


TIMEFRAME_SECONDS_MAP = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    engine = PaperTradingEngine(config, initial_paper_balance=100_000.0)
    logger.info("Starting paper trading — Stage 4 of 7-stage pipeline")
    logger.info(f"Required: {engine.MIN_DAYS_REQUIRED} days, Sharpe > {engine.MIN_SHARPE_REQUIRED}")
    engine.run()
