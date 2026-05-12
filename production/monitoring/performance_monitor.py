"""
performance_monitor.py — Real-Time Performance Monitor
Tracks live trading performance, computes statistics, and provides
data for the monitoring dashboard.

Metrics computed:
    - Equity curve and drawdown (real-time)
    - Rolling Sharpe, Sortino, Calmar ratios
    - Win rate, profit factor, average win/loss
    - Per-strategy attribution
    - Slippage cost tracking
    - Position-level unrealised P&L
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics."""
    timestamp: datetime

    # Account
    balance: float
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    daily_pnl_pct: float

    # Return statistics (based on daily P&L history)
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_hold_hours: float

    # Cost tracking
    total_slippage_usd: float
    avg_slippage_pts: float

    # Days trading
    trading_days: int

    def log_summary(self) -> str:
        return (
            f"Equity=${self.equity:,.2f} | DD={self.drawdown_pct:.1%} | "
            f"SR={self.sharpe_ratio:.2f} | WR={self.win_rate:.1%} | "
            f"PF={self.profit_factor:.2f} | trades={self.n_trades}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────
class PerformanceMonitor:
    """
    Real-time performance tracking for the trading system.
    Updated on every trade close and every equity poll.
    """

    def __init__(self, initial_balance: float, config: "SystemConfig"):
        self._initial_balance = initial_balance
        self._config = config

        # Equity tracking
        self._equity_history: list[tuple[datetime, float]] = [
            (datetime.now(timezone.utc), initial_balance)
        ]
        self._peak_equity = initial_balance
        self._daily_equity: dict[date, float] = {}
        self._daily_start_equity = initial_balance

        # Trade records
        self._closed_trades: list[dict] = []
        self._strategy_pnl: dict[str, list[float]] = {}

        # Slippage
        self._slippage_records: list[float] = []

        self._trading_day_start: Optional[date] = None

    # ── Equity updates ────────────────────────────────────────────────────────
    def update_equity(self, equity: float) -> None:
        """Record current equity value. Call every monitoring interval."""
        now = datetime.now(timezone.utc)
        today = now.date()

        self._equity_history.append((now, equity))

        # Peak tracking
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Daily tracking
        if today not in self._daily_equity:
            self._daily_equity[today] = equity
            self._daily_start_equity = equity
            if self._trading_day_start is None:
                self._trading_day_start = today

    def record_trade_close(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        volume: float,
        entry_price: float,
        close_price: float,
        pnl: float,
        hold_hours: float,
        slippage_pts: float = 0.0,
    ) -> None:
        """Record a closed trade for statistics."""
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "close_price": close_price,
            "pnl": pnl,
            "hold_hours": hold_hours,
            "slippage_pts": slippage_pts,
            "is_win": pnl > 0,
        }
        self._closed_trades.append(trade)

        if strategy not in self._strategy_pnl:
            self._strategy_pnl[strategy] = []
        self._strategy_pnl[strategy].append(pnl)

        self._slippage_records.append(abs(slippage_pts))

        logger.debug(
            f"Trade recorded: {symbol} {strategy} pnl={pnl:.2f} "
            f"hold={hold_hours:.1f}h slip={slippage_pts:.1f}pts"
        )

    # ── Snapshot computation ──────────────────────────────────────────────────
    def get_snapshot(
        self,
        current_equity: float,
        current_balance: float,
    ) -> PerformanceSnapshot:
        """Compute and return current performance snapshot."""
        now = datetime.now(timezone.utc)
        today = now.date()

        # Daily P&L
        daily_start = self._daily_equity.get(today, self._daily_start_equity)
        daily_pnl = current_equity - daily_start
        daily_pnl_pct = daily_pnl / (daily_start + 1e-10)

        # Drawdown
        drawdown_pct = max(0.0, (self._peak_equity - current_equity) / (self._peak_equity + 1e-10))

        # Total return
        total_return = (current_equity - self._initial_balance) / (self._initial_balance + 1e-10)

        # Daily returns series for ratio computation
        daily_returns = self._compute_daily_returns()
        trading_days = len(daily_returns)

        annualised_return = (
            float(np.mean(daily_returns) * 252)
            if len(daily_returns) > 0 else 0.0
        )
        sharpe = self._compute_sharpe(daily_returns)
        sortino = self._compute_sortino(daily_returns)
        calmar = self._compute_calmar(annualised_return, drawdown_pct)

        # Trade statistics
        n_trades = len(self._closed_trades)
        wins = [t for t in self._closed_trades if t["is_win"]]
        losses = [t for t in self._closed_trades if not t["is_win"]]

        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / max(n_trades, 1)
        avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        profit_factor = (
            abs(sum(t["pnl"] for t in wins)) /
            (abs(sum(t["pnl"] for t in losses)) + 1e-10)
            if losses else float("inf")
        )
        avg_hold = (
            float(np.mean([t["hold_hours"] for t in self._closed_trades]))
            if self._closed_trades else 0.0
        )

        # Slippage costs
        total_slip = (
            float(np.sum(self._slippage_records))
            if self._slippage_records else 0.0
        )
        avg_slip = (
            float(np.mean(self._slippage_records))
            if self._slippage_records else 0.0
        )

        return PerformanceSnapshot(
            timestamp=now,
            balance=current_balance,
            equity=current_equity,
            peak_equity=self._peak_equity,
            drawdown_pct=drawdown_pct,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            total_return_pct=total_return,
            annualised_return_pct=annualised_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_hold_hours=avg_hold,
            total_slippage_usd=total_slip,
            avg_slippage_pts=avg_slip,
            trading_days=trading_days,
        )

    # ── Per-strategy attribution ──────────────────────────────────────────────
    def strategy_attribution(self) -> dict[str, dict]:
        """Return P&L attribution per strategy."""
        result = {}
        for strategy, pnls in self._strategy_pnl.items():
            arr = np.array(pnls)
            n = len(arr)
            wins = int(np.sum(arr > 0))
            result[strategy] = {
                "n_trades": n,
                "total_pnl": float(np.sum(arr)),
                "avg_pnl": float(np.mean(arr)),
                "win_rate": wins / max(n, 1),
                "sharpe": (
                    float(np.mean(arr) / np.std(arr) * np.sqrt(252))
                    if n > 1 and np.std(arr) > 1e-10 else 0.0
                ),
            }
        return result

    # ── Ratio helpers ─────────────────────────────────────────────────────────
    def _compute_daily_returns(self) -> np.ndarray:
        """Build daily return series from equity history."""
        if len(self._daily_equity) < 2:
            return np.array([])

        sorted_days = sorted(self._daily_equity.items())
        equities = [v for _, v in sorted_days]
        returns = np.diff(equities) / (np.array(equities[:-1]) + 1e-10)
        return returns

    def _compute_sharpe(
        self,
        daily_returns: np.ndarray,
        risk_free_daily: float = 0.0,
    ) -> float:
        if len(daily_returns) < 5:
            return 0.0
        excess = daily_returns - risk_free_daily
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    def _compute_sortino(
        self,
        daily_returns: np.ndarray,
        target: float = 0.0,
    ) -> float:
        """Sortino ratio uses downside deviation only."""
        if len(daily_returns) < 5:
            return 0.0
        downside = daily_returns[daily_returns < target]
        if len(downside) < 2:
            return 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std < 1e-10:
            return 0.0
        return float(np.mean(daily_returns - target) / downside_std * np.sqrt(252))

    def _compute_calmar(
        self,
        annualised_return: float,
        max_drawdown: float,
    ) -> float:
        """Calmar = annualised return / max drawdown."""
        if max_drawdown < 1e-10:
            return 0.0
        return float(annualised_return / max_drawdown)

    def _compute_max_drawdown(self) -> float:
        """Maximum historical drawdown from equity curve."""
        if len(self._equity_history) < 2:
            return 0.0
        equities = np.array([e for _, e in self._equity_history])
        peak = np.maximum.accumulate(equities)
        drawdowns = (peak - equities) / (peak + 1e-10)
        return float(np.max(drawdowns))

    # ── Structured log export ─────────────────────────────────────────────────
    def export_daily_summary(self, current_equity: float, current_balance: float) -> dict:
        """Export end-of-day summary for structured JSON log."""
        snap = self.get_snapshot(current_equity, current_balance)
        return {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "equity": snap.equity,
            "balance": snap.balance,
            "daily_pnl": snap.daily_pnl,
            "daily_pnl_pct": snap.daily_pnl_pct,
            "drawdown_pct": snap.drawdown_pct,
            "sharpe_ytd": snap.sharpe_ratio,
            "n_trades_today": sum(
                1 for t in self._closed_trades
                if t["timestamp"][:10] == datetime.now(timezone.utc).date().isoformat()
            ),
            "win_rate_all": snap.win_rate,
            "profit_factor_all": snap.profit_factor,
            "avg_slippage_pts": snap.avg_slippage_pts,
            "strategy_attribution": self.strategy_attribution(),
        }
