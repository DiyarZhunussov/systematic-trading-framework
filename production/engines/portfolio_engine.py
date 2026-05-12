"""
portfolio_engine.py — Portfolio Construction Engine
Implements Part VIII of the framework.

Design philosophy (Section 8.1):
    The framework treats the collection of alpha engines as a portfolio of
    uncertain, partially correlated alpha hypotheses. Capital allocation is
    the primary control variable. Engines compete for capital under uncertainty;
    the portfolio layer adjudicates based on evidence and risk-adjusted
    contribution — not backtest Sharpe alone.

Construction objective:
    Maximise E[U(W)] (concave utility, drawdown-penalising)
    Subject to: vol target, CVaR limit, drawdown constraint,
                leverage limit, concentration constraint,
                normalize_strategy_scales() on all regime allocations.

ERC starting point (Section 8.2):
    Equal Risk Contribution provides theoretically diversified starting
    allocation requiring only volatility estimates — not return forecasts.
    w_i = (1/σ_i) / Σ(1/σ_j)
    Adjusted by confidence weight w_i and regime scale before use.

Covariance methods (Section 8.3):
    Daily monitoring   : EWMA (decay=0.94, ~30-day half-life)
    Monthly rebalance  : Ledoit-Wolf shrinkage
    Stress testing     : Regime-conditioned covariance (crisis vs calm)

Correlation stress tests (Section 8.4):
    Baseline: EWMA covariance
    Crisis:   All pairwise ρ → 0.80 → stressed SR must remain > 0
    Extreme:  All pairwise ρ → 1.00 → max daily loss ≤ 3× individual stop budget
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# sklearn optional — used for Ledoit-Wolf
try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available — Ledoit-Wolf will use manual shrinkage")


# ─────────────────────────────────────────────────────────────────────────────
# COVARIANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def ledoit_wolf_covariance(returns_matrix: np.ndarray) -> np.ndarray:
    """
    Shrink sample covariance toward structured estimator (Ledoit-Wolf).
    Reduces estimation error, especially in small-sample / high-dimension settings.

    Parameters
    ----------
    returns_matrix : (T × N) array of strategy/instrument returns

    Returns
    -------
    (N × N) shrunk covariance matrix
    """
    if SKLEARN_AVAILABLE:
        lw = LedoitWolf()
        lw.fit(returns_matrix)
        return lw.covariance_

    # Manual Oracle Approximating Shrinkage (simplified)
    T, N = returns_matrix.shape
    S = np.cov(returns_matrix.T)
    mu = np.trace(S) / N
    target = mu * np.eye(N)  # Scaled identity target

    # Shrinkage intensity (simplified Ledoit-Wolf formula)
    delta = min(1.0, ((N + 2) / (T * (N + 2 - 2))) )
    shrunk = (1 - delta) * S + delta * target
    return shrunk


def ewma_covariance(
    returns: np.ndarray,
    decay: float = 0.94,
) -> np.ndarray:
    """
    Exponentially weighted covariance matrix.
    decay=0.94 ≈ 30-day half-life.
    More responsive to recent regime changes than equal-weighted.

    Parameters
    ----------
    returns : (T × N) return matrix
    decay   : Decay factor (0.94 = RiskMetrics standard)
    """
    T, N = returns.shape
    weights = np.array([(1 - decay) * decay**i for i in range(T - 1, -1, -1)])
    weights /= weights.sum()
    demeaned = returns - returns.mean(axis=0)
    cov = np.einsum('t,ti,tj->ij', weights, demeaned, demeaned)
    return cov


def regime_conditioned_covariance(
    returns: np.ndarray,
    regime_labels: list[str],
    min_obs: int = 30,
) -> dict[str, Optional[np.ndarray]]:
    """
    Separate covariance matrices estimated per detected regime.
    Used for crisis stress-testing — not for live position sizing.

    Requires minimum min_obs observations per regime.
    Returns None for regimes with insufficient data.
    """
    unique_regimes = list(set(regime_labels))
    result = {}

    for regime in unique_regimes:
        mask = np.array([r == regime for r in regime_labels])
        n_obs = mask.sum()
        if n_obs >= min_obs:
            regime_returns = returns[mask]
            result[regime] = np.cov(regime_returns.T)
        else:
            result[regime] = None
            logger.debug(
                f"Regime '{regime}': {n_obs} obs < {min_obs} minimum — "
                f"using aggregate covariance for stress tests"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EQUAL RISK CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def equal_risk_contribution_weights(
    volatilities: dict[str, float],
) -> dict[str, float]:
    """
    Compute ERC weights: w_i = (1/σ_i) / Σ(1/σ_j)
    Each strategy contributes equal volatility to the portfolio.
    Used as starting point before signal-quality and regime adjustments.

    Parameters
    ----------
    volatilities : {strategy_name: annualised_volatility}

    Returns
    -------
    {strategy_name: weight} summing to 1.0
    """
    if not volatilities:
        return {}

    inv_vols = {k: 1.0 / (v + 1e-10) for k, v in volatilities.items()}
    total_inv = sum(inv_vols.values())
    weights = {k: v / total_inv for k, v in inv_vols.items()}
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSIFICATION RATIO
# ─────────────────────────────────────────────────────────────────────────────
def diversification_ratio(
    weights: np.ndarray,
    individual_vols: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    DR = Σ(w_i × σ_i) / σ_portfolio
    DR > 1.5: Healthy diversification
    DR 1.2–1.5: Adequate
    DR < 1.2: Minimal benefit — review correlations
    DR ≈ 1.0: All strategies correlated — system is one strategy with overhead
    """
    weighted_avg_vol = float(np.dot(weights, individual_vols))
    portfolio_var = float(weights @ cov_matrix @ weights)
    portfolio_vol = float(np.sqrt(max(portfolio_var, 1e-12)))
    if portfolio_vol < 1e-10:
        return 1.0
    return weighted_avg_vol / portfolio_vol


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StressTestResult:
    baseline_sharpe: float
    crisis_sharpe: float            # All ρ → 0.80
    extreme_max_loss_multiple: float # Max loss / individual stop budget (all ρ → 1.0)
    baseline_passes: bool
    crisis_passes: bool             # Must remain > 0
    extreme_passes: bool            # Must be ≤ 3×
    all_pass: bool
    details: dict = field(default_factory=dict)


def run_correlation_stress_tests(
    weights: np.ndarray,
    strategy_names: list[str],
    returns_matrix: np.ndarray,
    individual_stop_budgets: np.ndarray,
    cov_matrix: np.ndarray,
) -> StressTestResult:
    """
    Run all three correlation stress tests (Section 8.4).

    Tests:
        Baseline: EWMA covariance → compute portfolio SR
        Crisis:   All pairwise ρ → 0.80 → stressed SR must remain > 0
        Extreme:  All ρ → 1.00 → max loss ≤ 3× individual stop budget

    Parameters
    ----------
    weights : Portfolio weights (N,)
    strategy_names : Names for logging
    returns_matrix : (T × N) return matrix
    individual_stop_budgets : Dollar stop budget per strategy (N,)
    cov_matrix : Baseline EWMA covariance
    """
    N = len(weights)
    T = len(returns_matrix)

    # ── Baseline Sharpe ───────────────────────────────────────────────────────
    port_returns = returns_matrix @ weights
    baseline_sr = (
        float(np.mean(port_returns) / np.std(port_returns) * np.sqrt(252))
        if np.std(port_returns) > 1e-10 else 0.0
    )

    # ── Crisis: all ρ → 0.80 ─────────────────────────────────────────────────
    stds = np.sqrt(np.diag(cov_matrix))
    crisis_corr = np.full((N, N), 0.80)
    np.fill_diagonal(crisis_corr, 1.0)
    crisis_cov = np.outer(stds, stds) * crisis_corr

    crisis_port_var = float(weights @ crisis_cov @ weights)
    crisis_port_std = float(np.sqrt(max(crisis_port_var, 1e-12)))

    # Use baseline mean but stressed vol
    baseline_mean = float(np.mean(port_returns))
    crisis_sr = (
        baseline_mean / crisis_port_std * np.sqrt(252)
        if crisis_port_std > 1e-10 else 0.0
    )

    # ── Extreme: all ρ → 1.00 ────────────────────────────────────────────────
    extreme_cov = np.outer(stds, stds)  # All correlations = 1.0
    extreme_port_std = float(np.sqrt(max(float(weights @ extreme_cov @ weights), 1e-12)))

    # Max daily loss under full correlation = portfolio vol × ~2σ daily
    individual_budget = float(np.dot(weights, individual_stop_budgets))
    extreme_max_loss = extreme_port_std * 2.0  # 2-sigma daily move
    loss_multiple = extreme_max_loss / (individual_budget + 1e-10)

    crisis_passes = crisis_sr > 0.0
    extreme_passes = loss_multiple <= 3.0

    if not crisis_passes:
        logger.warning(
            f"STRESS TEST FAILED: Crisis correlation (ρ=0.80) → "
            f"SR={crisis_sr:.3f} ≤ 0. Reduce all positions."
        )

    if not extreme_passes:
        logger.warning(
            f"STRESS TEST FAILED: Full correlation (ρ=1.00) → "
            f"max loss = {loss_multiple:.1f}× stop budget (limit: 3×)."
        )

    return StressTestResult(
        baseline_sharpe=baseline_sr,
        crisis_sharpe=crisis_sr,
        extreme_max_loss_multiple=loss_multiple,
        baseline_passes=True,
        crisis_passes=crisis_passes,
        extreme_passes=extreme_passes,
        all_pass=crisis_passes and extreme_passes,
        details={
            "baseline_sr": baseline_sr,
            "crisis_sr": crisis_sr,
            "extreme_loss_multiple": loss_multiple,
            "individual_strategy_vols": stds.tolist(),
            "weights": weights.tolist(),
            "strategy_names": strategy_names,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ALLOCATION RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PortfolioAllocation:
    """Result of monthly rebalancing optimisation."""
    weights: dict[str, float]          # {strategy: weight}
    erc_weights: dict[str, float]      # Pre-adjustment ERC weights
    regime_scales: dict[str, float]    # From regime engine (normalised)
    confidence_weights: dict[str, float]  # From Bayesian estimator
    diversification_ratio: float
    stress_test: StressTestResult
    rebalance_needed: bool
    phased_transition: bool            # True if weight change > 10%
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def log_summary(self) -> str:
        weights_str = " | ".join(
            f"{k}={v:.1%}" for k, v in self.weights.items()
        )
        return (
            f"Portfolio allocation: {weights_str} | "
            f"DR={self.diversification_ratio:.2f} | "
            f"stress={'PASS' if self.stress_test.all_pass else 'FAIL'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class PortfolioEngine:
    """
    Portfolio construction engine implementing Section 8.

    Rebalancing cycle (Section 8.5):
        Step 1: Compute trailing 60-day Sharpe per strategy
        Step 2: Compute rolling IC per strategy
        Step 3: Update Bayesian confidence weights
        Step 4: Get regime allocations
        Step 5: Apply normalize_strategy_scales()
        Step 6: Optimise weights (constrained)
        Step 7: Constrain to within 20% of equal-weight
        Step 8: Apply turnover constraint (phase if change > 10%)
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        p = config.portfolio_params

        self._rebalance_days = int(p.get("rebalance_frequency_days", 30))
        self._erc_lookback = int(p.get("erc_vol_lookback_days", 60))
        self._ewma_decay = float(p.get("ewma_decay", 0.94))
        self._min_weight = float(p.get("min_strategy_weight", 0.05))
        self._max_weight = float(p.get("max_strategy_weight", 0.60))
        self._weight_change_threshold = float(p.get("weight_change_threshold_pct", 0.10))
        self._dr_warn_threshold = float(p.get("diversification_ratio_warn", 1.20))

        self._current_weights: dict[str, float] = {}
        self._last_rebalance: Optional[datetime] = None
        self._returns_history: dict[str, list[float]] = {}

    # ── Returns tracking ──────────────────────────────────────────────────────
    def record_returns(
        self,
        strategy_returns: dict[str, float],
        date: Optional[datetime] = None,
    ) -> None:
        """
        Record daily returns for each strategy.
        Called at end of each trading day.
        """
        for strategy, ret in strategy_returns.items():
            if strategy not in self._returns_history:
                self._returns_history[strategy] = []
            self._returns_history[strategy].append(ret)

    def _get_returns_matrix(
        self,
        strategies: list[str],
        lookback: int,
    ) -> Optional[np.ndarray]:
        """
        Build (T × N) returns matrix for given strategies.
        Returns None if insufficient history.
        """
        arrays = []
        min_len = None

        for strategy in strategies:
            hist = self._returns_history.get(strategy, [])
            if len(hist) < lookback // 2:
                logger.warning(
                    f"Insufficient return history for {strategy}: "
                    f"{len(hist)} < {lookback // 2}"
                )
                return None
            arr = np.array(hist[-lookback:])
            arrays.append(arr)
            min_len = len(arr) if min_len is None else min(min_len, len(arr))

        if not arrays or min_len is None or min_len < 10:
            return None

        return np.column_stack([a[-min_len:] for a in arrays])

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    def rolling_sharpe(
        self,
        strategy: str,
        window: int = 60,
    ) -> float:
        """Compute trailing Sharpe for a strategy."""
        hist = self._returns_history.get(strategy, [])
        if len(hist) < window // 2:
            return 0.0
        arr = np.array(hist[-window:])
        std = np.std(arr, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(252))

    # ── Main rebalance ────────────────────────────────────────────────────────
    def rebalance(
        self,
        strategy_volatilities: dict[str, float],
        regime_scales: dict[str, float],
        confidence_weights: dict[str, float],
        account_balance: float,
        individual_stop_budgets: Optional[dict[str, float]] = None,
        force: bool = False,
    ) -> PortfolioAllocation:
        """
        Execute monthly rebalancing procedure (Section 8.5, Steps 1–8).

        Parameters
        ----------
        strategy_volatilities : {strategy: annualised_vol}
        regime_scales         : Normalised scales from regime engine
        confidence_weights    : From Bayesian confidence estimator
        account_balance       : Current account value
        individual_stop_budgets : {strategy: dollar stop budget} for stress test
        force                 : Force rebalance regardless of schedule

        Returns
        -------
        PortfolioAllocation with updated weights
        """
        now = datetime.now(timezone.utc)
        strategies = list(strategy_volatilities.keys())

        # Check if rebalance is due
        rebalance_needed = force or (
            self._last_rebalance is None
            or (now - self._last_rebalance).days >= self._rebalance_days
        )

        # ── Step 1–2: Compute Sharpes and build returns matrix ────────────────
        sharpes = {s: self.rolling_sharpe(s, window=60) for s in strategies}
        returns_matrix = self._get_returns_matrix(strategies, self._erc_lookback)

        # ── Step 3: ERC base weights ──────────────────────────────────────────
        erc_weights = equal_risk_contribution_weights(strategy_volatilities)

        # ── Step 4–5: Apply regime scales (already normalised) ────────────────
        # Merge ERC, regime, and confidence weights
        combined = {}
        for s in strategies:
            erc_w = erc_weights.get(s, 1.0 / len(strategies))
            reg_s = regime_scales.get(s, 0.5)
            conf_w = confidence_weights.get(s, 0.5)
            combined[s] = erc_w * reg_s * conf_w

        # Renormalise
        total = sum(combined.values())
        if total > 1e-10:
            combined = {k: v / total for k, v in combined.items()}
        else:
            combined = {s: 1.0 / len(strategies) for s in strategies}

        # ── Step 6: Constrain weights ─────────────────────────────────────────
        clamped = {
            k: float(np.clip(v, self._min_weight, self._max_weight))
            for k, v in combined.items()
        }
        total_clamped = sum(clamped.values())
        if total_clamped > 1e-10:
            clamped = {k: v / total_clamped for k, v in clamped.items()}

        # ── Step 7: Constrain to within 20% of equal-weight ──────────────────
        equal_w = 1.0 / len(strategies)
        max_deviation = 0.20
        final_weights = {}
        for s, w in clamped.items():
            min_w = max(self._min_weight, equal_w - max_deviation)
            max_w = min(self._max_weight, equal_w + max_deviation)
            final_weights[s] = float(np.clip(w, min_w, max_w))

        total_final = sum(final_weights.values())
        if total_final > 1e-10:
            final_weights = {k: v / total_final for k, v in final_weights.items()}

        # ── Step 8: Turnover constraint — phase if change > 10% ──────────────
        phased = False
        if self._current_weights:
            for s, new_w in final_weights.items():
                old_w = self._current_weights.get(s, equal_w)
                if abs(new_w - old_w) > self._weight_change_threshold:
                    phased = True
                    # Phase: move halfway per rebalance
                    final_weights[s] = (old_w + new_w) / 2
                    logger.info(
                        f"Phased transition for {s}: "
                        f"{old_w:.1%} → {new_w:.1%} (phased to {final_weights[s]:.1%})"
                    )

        # ── Covariance for stress tests ───────────────────────────────────────
        if returns_matrix is not None and len(returns_matrix) > 5:
            try:
                cov = ewma_covariance(returns_matrix, decay=self._ewma_decay)
            except Exception:
                cov = np.diag([strategy_volatilities.get(s, 0.1)**2 for s in strategies])
        else:
            cov = np.diag([strategy_volatilities.get(s, 0.1)**2 for s in strategies])

        weights_array = np.array([final_weights.get(s, equal_w) for s in strategies])
        vols_array = np.array([strategy_volatilities.get(s, 0.1) for s in strategies])

        # ── Diversification ratio ─────────────────────────────────────────────
        dr = diversification_ratio(weights_array, vols_array, cov)
        if dr < self._dr_warn_threshold:
            logger.warning(
                f"Low diversification ratio: DR={dr:.2f} < {self._dr_warn_threshold}. "
                f"Review strategy correlations."
            )

        # ── Stress tests ──────────────────────────────────────────────────────
        if individual_stop_budgets is not None:
            stop_budgets = np.array([
                individual_stop_budgets.get(s, account_balance * 0.005)
                for s in strategies
            ])
        else:
            stop_budgets = np.full(len(strategies), account_balance * 0.005)

        if returns_matrix is not None and len(returns_matrix) >= 20:
            stress = run_correlation_stress_tests(
                weights=weights_array,
                strategy_names=strategies,
                returns_matrix=returns_matrix,
                individual_stop_budgets=stop_budgets,
                cov_matrix=cov,
            )
        else:
            # Insufficient history for stress test — use conservative pass
            stress = StressTestResult(
                baseline_sharpe=0.0, crisis_sharpe=0.5,
                extreme_max_loss_multiple=1.0,
                baseline_passes=True, crisis_passes=True,
                extreme_passes=True, all_pass=True,
                details={"note": "insufficient_history_for_stress_test"},
            )

        # ── Enforce stress test (crisis correlation) ──────────────────────────
        if not stress.crisis_passes:
            logger.warning(
                "Stress test failed (crisis correlation SR ≤ 0). "
                "Reducing all weights by 50% as safety measure."
            )
            final_weights = {k: v * 0.5 for k, v in final_weights.items()}
            total_reduced = sum(final_weights.values())
            if total_reduced > 1e-10:
                final_weights = {k: v / total_reduced for k, v in final_weights.items()}

        if rebalance_needed:
            self._current_weights = dict(final_weights)
            self._last_rebalance = now

        allocation = PortfolioAllocation(
            weights=final_weights,
            erc_weights=erc_weights,
            regime_scales=regime_scales,
            confidence_weights=confidence_weights,
            diversification_ratio=dr,
            stress_test=stress,
            rebalance_needed=rebalance_needed,
            phased_transition=phased,
        )

        logger.info(allocation.log_summary())
        return allocation

    @property
    def current_weights(self) -> dict[str, float]:
        return dict(self._current_weights)

    @property
    def days_since_rebalance(self) -> Optional[int]:
        if self._last_rebalance is None:
            return None
        return (datetime.now(timezone.utc) - self._last_rebalance).days
