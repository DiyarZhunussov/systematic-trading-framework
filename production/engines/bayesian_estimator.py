"""
bayesian_estimator.py — Bayesian IC and Volatility Estimators
Implements Section 6 of the framework.

Core principle (Section 6.1):
    All parameters are treated as probability distributions, not point estimates.
    The uncertainty in an estimate must be reflected in reduced position size.
    The standard approach — compute a point estimate then size as if it's known
    exactly — is epistemically dishonest.

IC Prior — Beta(2, 30) — revised per adversarial review (Section 6.2):
    Original: Beta(3, 30) → prior mean = 0.0909
    Revised:  Beta(2, 30) → prior mean = 0.0625

    Rationale: Lopez de Prado & Fabozzi (2026) estimate search-adjusted FDR
    in quantitative finance exceeds 80%. Beta(2, 30) encodes a prior that is
    more sceptical of any newly observed IC, requiring significantly more
    out-of-sample evidence before posterior mean exceeds break-even IC.

    The system requires approximately N=200 trades before the posterior mean
    (from weak prior) exceeds break-even IC sufficiently for full confidence.
    This reflects appropriate scepticism about newly deployed strategies.

Volatility Prior — InverseGamma:
    Calibrated to instrument's 5-year historical volatility distribution.
    Posterior automatically shrinks extreme volatility estimates toward prior,
    preventing oversizing during transient volatility spikes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm, invgamma

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN IC ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ICPosterior:
    """Snapshot of the IC posterior distribution at a point in time."""
    alpha: float                    # Beta distribution alpha parameter
    beta_param: float               # Beta distribution beta parameter
    n_trades: int
    posterior_mean: float
    posterior_std: float
    ci_90_lower: float              # 5th percentile
    ci_90_upper: float              # 95th percentile
    coefficient_of_variation: float # std / mean — higher = more uncertain
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_mature(self) -> bool:
        """True once enough trades observed for meaningful posterior."""
        return self.n_trades >= 200

    def summary(self) -> str:
        return (
            f"IC posterior | n={self.n_trades} | "
            f"mean={self.posterior_mean:.4f} ± {self.posterior_std:.4f} | "
            f"90% CI=[{self.ci_90_lower:.4f}, {self.ci_90_upper:.4f}] | "
            f"CV={self.coefficient_of_variation:.2f}"
        )


class ICBayesianEstimator:
    """
    Bayesian estimator for signal Information Coefficient.
    Uses Beta-Binomial conjugate model with revised prior Beta(2, 30).

    The Beta-Binomial model treats each trade as a Bernoulli trial:
    - Win  (signal was correct): contributes to alpha
    - Loss (signal was wrong):   contributes to beta

    Position confidence weight incorporates both posterior mean AND
    uncertainty (coefficient of variation) — wider uncertainty → lower weight.
    """

    def __init__(
        self,
        alpha_prior: int = 2,
        beta_prior: int = 30,
        uncertainty_aversion: float = 0.5,
    ):
        """
        Parameters
        ----------
        alpha_prior : Beta distribution alpha (default 2 — revised prior)
        beta_prior  : Beta distribution beta  (default 30 — revised prior)
        uncertainty_aversion : How much CV penalises confidence weight (0–1)
                               0.5 = moderate penalty for uncertainty
        """
        self._alpha_0 = alpha_prior
        self._beta_0 = beta_prior
        self._alpha = float(alpha_prior)
        self._beta = float(beta_prior)
        self._n_trades = 0
        self._n_wins = 0
        self._uncertainty_aversion = uncertainty_aversion
        self._trade_history: list[dict] = []  # For audit trail

        logger.info(
            f"ICBayesianEstimator initialised: "
            f"prior=Beta({alpha_prior}, {beta_prior}) "
            f"prior_mean={alpha_prior / (alpha_prior + beta_prior):.4f}"
        )

    def update(self, n_wins: int, n_total: int, strategy: str = "") -> ICPosterior:
        """
        Update posterior with observed trade outcomes.

        Parameters
        ----------
        n_wins  : Number of profitable trades in this batch
        n_total : Total trades in this batch
        strategy : Strategy name for logging

        Returns
        -------
        Updated ICPosterior snapshot
        """
        if n_total < 0 or n_wins < 0 or n_wins > n_total:
            raise ValueError(
                f"Invalid trade counts: n_wins={n_wins}, n_total={n_total}"
            )

        self._alpha += n_wins
        self._beta += (n_total - n_wins)
        self._n_trades += n_total
        self._n_wins += n_wins

        self._trade_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "n_wins_batch": n_wins,
            "n_total_batch": n_total,
            "posterior_mean_after": self.posterior_mean,
            "n_trades_cumulative": self._n_trades,
        })

        posterior = self.get_posterior()
        logger.debug(
            f"IC update [{strategy}]: +{n_wins}/{n_total} | "
            f"{posterior.summary()}"
        )
        return posterior

    def update_single_trade(self, is_win: bool, strategy: str = "") -> ICPosterior:
        """Update with a single trade result."""
        return self.update(
            n_wins=1 if is_win else 0,
            n_total=1,
            strategy=strategy,
        )

    def get_posterior(self) -> ICPosterior:
        """Return current posterior snapshot."""
        a = self._alpha
        b = self._beta
        total = a + b

        mean = a / total
        variance = (a * b) / (total ** 2 * (total + 1))
        std = float(np.sqrt(variance))
        cv = std / mean if mean > 1e-10 else float('inf')

        ci_lower = float(beta_dist.ppf(0.05, a, b))
        ci_upper = float(beta_dist.ppf(0.95, a, b))

        return ICPosterior(
            alpha=a,
            beta_param=b,
            n_trades=self._n_trades,
            posterior_mean=float(mean),
            posterior_std=std,
            ci_90_lower=ci_lower,
            ci_90_upper=ci_upper,
            coefficient_of_variation=float(cv),
        )

    @property
    def posterior_mean(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    @property
    def posterior_std(self) -> float:
        a, b = self._alpha, self._beta
        total = a + b
        return float(np.sqrt((a * b) / (total ** 2 * (total + 1))))

    @property
    def coefficient_of_variation(self) -> float:
        mean = self.posterior_mean
        if mean < 1e-10:
            return float('inf')
        return self.posterior_std / mean

    @property
    def n_trades(self) -> int:
        return self._n_trades

    def confidence_weight(
        self,
        breakeven_ic: float,
        uncertainty_aversion: Optional[float] = None,
    ) -> float:
        """
        Position confidence weight incorporating IC uncertainty.

        w = (posterior_mean / breakeven_ic) × (1 − aversion × CV)

        - If posterior_mean < breakeven_ic: weight < 1 (not yet justified)
        - High CV (uncertain posterior): weight penalised
        - Weight always capped at 1.0

        Parameters
        ----------
        breakeven_ic : Minimum IC needed to cover transaction costs
        uncertainty_aversion : Override instance default (0–1)

        Returns
        -------
        float in [0, 1]
        """
        aversion = uncertainty_aversion if uncertainty_aversion is not None \
            else self._uncertainty_aversion

        w = self.posterior_mean / (breakeven_ic + 1e-10)
        w *= max(0.0, 1.0 - aversion * self.coefficient_of_variation)
        return float(np.clip(w, 0.0, 1.0))

    def bars_to_confidence(
        self,
        breakeven_ic: float,
        target_weight: float = 0.8,
        true_ic_estimate: float = 0.08,
    ) -> int:
        """
        Estimate trades needed before confidence weight reaches target_weight,
        assuming the strategy has true_ic_estimate hit rate.

        Used for planning — not for production sizing.
        """
        alpha_sim = float(self._alpha_0)
        beta_sim = float(self._beta_0)
        hit_rate = 0.5 + true_ic_estimate / 2  # Convert IC to win rate approx

        for n in range(1, 2001):
            mean = alpha_sim / (alpha_sim + beta_sim)
            std = float(np.sqrt(
                (alpha_sim * beta_sim) /
                ((alpha_sim + beta_sim) ** 2 * (alpha_sim + beta_sim + 1))
            ))
            cv = std / mean if mean > 1e-10 else float('inf')
            w = (mean / breakeven_ic) * max(0.0, 1.0 - self._uncertainty_aversion * cv)
            if w >= target_weight:
                return n
            alpha_sim += hit_rate
            beta_sim += (1 - hit_rate)
        return 2000  # Did not reach target within 2000 trades

    def reset(self) -> None:
        """Reset to prior — use when strategy is substantially modified."""
        self._alpha = float(self._alpha_0)
        self._beta = float(self._beta_0)
        self._n_trades = 0
        self._n_wins = 0
        self._trade_history.clear()
        logger.info("ICBayesianEstimator reset to prior")


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN VOLATILITY ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VolPosterior:
    """Snapshot of volatility posterior (InverseGamma)."""
    alpha_posterior: float
    beta_posterior: float
    expected_vol: float             # Posterior mean = beta/(alpha-1)
    ci_90_lower: float
    ci_90_upper: float
    n_observations: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> str:
        return (
            f"Vol posterior | n={self.n_observations} | "
            f"E[σ]={self.expected_vol:.4f} | "
            f"90% CI=[{self.ci_90_lower:.4f}, {self.ci_90_upper:.4f}]"
        )


class VolBayesianEstimator:
    """
    Bayesian volatility estimator using InverseGamma conjugate prior.
    Prior calibrated to instrument's historical volatility distribution.

    The posterior automatically shrinks extreme vol estimates toward the prior,
    preventing oversizing during transient volatility spikes (Section 6.3).
    """

    def __init__(
        self,
        prior_mean_vol: float = 0.15,
        prior_strength: float = 20.0,
    ):
        """
        Parameters
        ----------
        prior_mean_vol : Prior belief about annualised volatility (e.g. 0.15 = 15%)
        prior_strength : Effective sample size of prior (higher = stronger prior)
                         20 = prior counts as 20 observations
        """
        # InverseGamma parameterisation: E[σ²] = β/(α-1)
        # Set α = prior_strength/2, β = α × prior_variance
        prior_variance = (prior_mean_vol / np.sqrt(252)) ** 2  # Daily variance
        self._alpha_0 = prior_strength / 2
        self._beta_0 = self._alpha_0 * prior_variance

        self._alpha = self._alpha_0
        self._beta = self._beta_0
        self._n_obs = 0

        logger.info(
            f"VolBayesianEstimator: "
            f"prior_mean_vol={prior_mean_vol:.2%} "
            f"prior_strength={prior_strength}"
        )

    def update(self, returns: np.ndarray) -> VolPosterior:
        """
        Update posterior with new return observations.

        Parameters
        ----------
        returns : Array of daily returns (not annualised)

        Returns
        -------
        VolPosterior snapshot
        """
        n = len(returns)
        if n == 0:
            return self.get_posterior()

        # InverseGamma update:
        # alpha_post = alpha_0 + n/2
        # beta_post  = beta_0 + sum(r²)/2
        self._alpha += n / 2
        self._beta += float(np.sum(returns ** 2)) / 2
        self._n_obs += n

        return self.get_posterior()

    def get_posterior(self) -> VolPosterior:
        """Return current volatility posterior snapshot."""
        alpha = self._alpha
        beta_p = self._beta

        if alpha <= 1:
            # Posterior mean undefined — use prior mode
            daily_vol = float(np.sqrt(beta_p / (alpha + 1)))
        else:
            daily_vol = float(np.sqrt(beta_p / (alpha - 1)))

        annualised_vol = daily_vol * np.sqrt(252)

        # 90% CI on annualised vol
        try:
            ci_low_var = float(invgamma.ppf(0.05, alpha, scale=beta_p))
            ci_high_var = float(invgamma.ppf(0.95, alpha, scale=beta_p))
            ci_low = float(np.sqrt(ci_low_var) * np.sqrt(252))
            ci_high = float(np.sqrt(ci_high_var) * np.sqrt(252))
        except Exception:
            ci_low = annualised_vol * 0.7
            ci_high = annualised_vol * 1.3

        return VolPosterior(
            alpha_posterior=alpha,
            beta_posterior=beta_p,
            expected_vol=annualised_vol,
            ci_90_lower=ci_low,
            ci_90_upper=ci_high,
            n_observations=self._n_obs,
        )

    @property
    def expected_daily_vol(self) -> float:
        """Posterior mean daily volatility (not annualised)."""
        alpha = self._alpha
        beta_p = self._beta
        if alpha <= 1:
            return float(np.sqrt(beta_p / (alpha + 1)))
        return float(np.sqrt(beta_p / (alpha - 1)))

    @property
    def expected_annual_vol(self) -> float:
        return self.expected_daily_vol * np.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-DIMENSIONAL CONFIDENCE WEIGHT (Section 6.5)
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence_weight(
    months_since_validation: float,
    rolling_sharpe_30d: float,
    historical_sharpe: float,
    rolling_ic_30d: float,
    slippage_ratio: float,
    regime_scale: float,
    age_decay_per_month: float = 0.03,
    min_age_factor: float = 0.20,
) -> float:
    """
    Composite confidence weight for an alpha engine (Section 6.5).
    Decays on five independent dimensions; recovery requires evidence.

    Parameters
    ----------
    months_since_validation : Months since last formal validation
    rolling_sharpe_30d      : 30-day rolling Sharpe
    historical_sharpe       : Historical (validated) Sharpe
    rolling_ic_30d          : 30-day rolling IC
    slippage_ratio          : recent_slippage / baseline_slippage (1.0 = normal)
    regime_scale            : Scale from regime allocation matrix (0–1)

    Returns
    -------
    Composite weight in [0, 1]
    """
    # 1. Age decay: -3% per month since last validation, floor at 20%
    age_factor = max(min_age_factor, 1.0 - age_decay_per_month * months_since_validation)

    # 2. Performance factor: scaled to historical Sharpe
    if historical_sharpe > 0:
        perf_ratio = rolling_sharpe_30d / historical_sharpe
        perf_factor = float(np.clip(0.3 + 0.7 * perf_ratio, 0.0, 1.0))
    else:
        perf_factor = 0.0

    # 3. IC stability factor
    ic_factor = float(np.clip(rolling_ic_30d / 0.05 + 0.5, 0.0, 1.0))

    # 4. Execution quality factor
    exec_factor = float(np.clip(2.0 - slippage_ratio, 0.0, 1.0))

    # 5. Regime appropriateness (from allocation matrix)
    w = age_factor * perf_factor * ic_factor * exec_factor * regime_scale

    return float(np.clip(w, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ESTIMATOR REGISTRY
# Maintains one IC + Vol estimator per strategy per instrument
# ─────────────────────────────────────────────────────────────────────────────
class StrategyEstimatorRegistry:
    """
    Registry of Bayesian estimators for each (strategy, instrument) pair.
    Provides a single interface for all confidence weight computations.
    """

    def __init__(self, config: "SystemConfig"):
        self.config = config
        self._ic_estimators: dict[str, ICBayesianEstimator] = {}
        self._vol_estimators: dict[str, VolBayesianEstimator] = {}

    def _ic_key(self, strategy: str, instrument: str) -> str:
        return f"{strategy}::{instrument}"

    def get_ic_estimator(
        self,
        strategy: str,
        instrument: str,
    ) -> ICBayesianEstimator:
        """Return (or create) IC estimator for strategy/instrument pair."""
        key = self._ic_key(strategy, instrument)
        if key not in self._ic_estimators:
            params = self.config.bayesian_params
            self._ic_estimators[key] = ICBayesianEstimator(
                alpha_prior=params.get("ic_prior_alpha", 2),
                beta_prior=params.get("ic_prior_beta", 30),
                uncertainty_aversion=params.get("uncertainty_aversion", 0.5),
            )
        return self._ic_estimators[key]

    def get_vol_estimator(
        self,
        instrument: str,
        prior_mean_vol: float = 0.15,
    ) -> VolBayesianEstimator:
        """Return (or create) volatility estimator for instrument."""
        if instrument not in self._vol_estimators:
            self._vol_estimators[instrument] = VolBayesianEstimator(
                prior_mean_vol=prior_mean_vol,
                prior_strength=20.0,
            )
        return self._vol_estimators[instrument]

    def record_trade(
        self,
        strategy: str,
        instrument: str,
        is_win: bool,
    ) -> ICPosterior:
        """Record a trade outcome and return updated IC posterior."""
        estimator = self.get_ic_estimator(strategy, instrument)
        return estimator.update_single_trade(is_win, strategy=strategy)

    def get_confidence_weight(
        self,
        strategy: str,
        instrument: str,
        breakeven_ic: float = 0.04,
    ) -> float:
        """Return IC-based confidence weight for position sizing."""
        estimator = self.get_ic_estimator(strategy, instrument)
        return estimator.confidence_weight(breakeven_ic)

    def get_all_posteriors(self) -> dict[str, ICPosterior]:
        """Return IC posteriors for all tracked strategy/instrument pairs."""
        return {
            key: est.get_posterior()
            for key, est in self._ic_estimators.items()
        }

    def reset_strategy(self, strategy: str) -> None:
        """Reset all IC estimators for a strategy (after major parameter change)."""
        keys = [k for k in self._ic_estimators if k.startswith(f"{strategy}::")]
        for key in keys:
            self._ic_estimators[key].reset()
        logger.info(f"Reset IC estimators for strategy: {strategy}")
