"""
deflated_sharpe.py — Deflated Sharpe Ratio (DSR)
Implements Section 5.3 of the framework exactly.

Reference: Bailey and López de Prado (2014)

The DSR corrects for three simultaneous sources of bias:
    1. Selection bias from multiple testing (N_trials parameter)
    2. Non-normality of returns (skewness and kurtosis correction)
    3. Finite track record length (sqrt((T-1)/T) correction)

Formula:
    SR* = SR × sqrt[(T−1)/T] / sqrt[1 − γ₃ × SR + (γ₄ − 1)/4 × SR²]

    Expected maximum Sharpe under N independent trials:
    SR_benchmark ≈ (1 − γ_E) × Φ⁻¹(1 − 1/N) + γ_E × Φ⁻¹(1 − 1/(N×e))
    where γ_E = 0.5772 (Euler-Mascheroni constant)

    DSR = Φ[(SR* − SR_benchmark) × sqrt(T−1)]

Acceptance thresholds (Section 5.3):
    DSR > 0.95 : statistically significant edge after selection bias correction
    DSR 0.85–0.95 : borderline — require additional OOS validation
    DSR < 0.85 : likely false discovery — do not deploy

Note on N_trials:
    For correlated variants (e.g. varying window 20→100 in steps of 5),
    effective N is lower than the raw count. Use López de Prado's ONC
    algorithm or conservatively use the full count of parameter combinations.
    When in doubt, use the full count — conservative but avoids
    underestimating the multiple-testing problem.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant
EULER_GAMMA = 0.5772156649


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DSRResult:
    """Full output of the Deflated Sharpe Ratio computation."""
    dsr: float                      # Probability SR > 0 after selection bias
    sr_observed: float              # Input observed Sharpe
    sr_star: float                  # Non-normality + finite-sample corrected SR
    sr_benchmark: float             # Expected max SR under pure noise
    t_observations: int
    skewness: float
    excess_kurtosis: float
    n_trials: int

    # Verdict
    passes_95: bool                 # DSR > 0.95
    passes_85: bool                 # DSR > 0.85
    verdict: str

    def log_summary(self) -> str:
        return (
            f"DSR={self.dsr:.4f} | SR_obs={self.sr_observed:.3f} | "
            f"SR*={self.sr_star:.3f} | SR_bench={self.sr_benchmark:.3f} | "
            f"T={self.t_observations} | N_trials={self.n_trials} | "
            f"verdict={self.verdict}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CORE DSR FUNCTION (exact implementation from PDF Section 5.3)
# ─────────────────────────────────────────────────────────────────────────────
def deflated_sharpe_ratio(
    sr_observed: float,
    t: int,
    skew: float,
    kurt: float,
    n_trials: int,
) -> DSRResult:
    """
    Compute Deflated Sharpe Ratio correcting for selection bias,
    non-normality, and track record length.

    Parameters
    ----------
    sr_observed : float — Sharpe ratio of selected strategy (annualised)
    t           : int   — number of return observations
    skew        : float — skewness of strategy daily returns
    kurt        : float — excess kurtosis of strategy daily returns
    n_trials    : int   — total number of strategies/parameter sets tested

    Returns
    -------
    DSRResult with full diagnostic breakdown

    Example (from PDF):
        strategy found by testing 50 variants, 3-year daily data
        dsr = deflated_sharpe_ratio(SR_observed=0.80, T=756, skew=0.15,
                                     kurt=1.20, N_trials=50)
        → DSR ≈ 0.64 (edge survives selection bias correction at moderate confidence)
    """
    if t < 5:
        raise ValueError(f"T={t} too small for reliable DSR computation (minimum 5)")
    if n_trials < 1:
        raise ValueError(f"N_trials={n_trials} must be ≥ 1")

    # ── Expected maximum Sharpe under pure noise (N independent trials) ───────
    sr_benchmark = (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # ── Non-normality and finite-sample correction ────────────────────────────
    # SR* = SR × sqrt[(T-1)/T] / sqrt[1 - skew×SR + (kurt-1)/4 × SR²]
    denominator = 1 - skew * sr_observed + (kurt - 1) / 4 * sr_observed ** 2

    if denominator <= 0:
        logger.warning(
            f"DSR: negative denominator ({denominator:.4f}) — "
            f"extreme skew/kurtosis values. Using |denominator|."
        )
        denominator = abs(denominator) + 1e-10

    sr_star = sr_observed * np.sqrt((t - 1) / t) / np.sqrt(denominator)

    # ── DSR: probability that SR > 0 after correction ─────────────────────────
    dsr = float(norm.cdf((sr_star - sr_benchmark) * np.sqrt(t - 1)))
    dsr = float(np.clip(dsr, 0.0, 1.0))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if dsr > 0.95:
        verdict = "DEPLOY_CANDIDATE"
    elif dsr > 0.85:
        verdict = "BORDERLINE_ADDITIONAL_OOS_REQUIRED"
    else:
        verdict = "LIKELY_FALSE_DISCOVERY_DO_NOT_DEPLOY"

    result = DSRResult(
        dsr=dsr,
        sr_observed=float(sr_observed),
        sr_star=float(sr_star),
        sr_benchmark=float(sr_benchmark),
        t_observations=int(t),
        skewness=float(skew),
        excess_kurtosis=float(kurt),
        n_trials=int(n_trials),
        passes_95=dsr > 0.95,
        passes_85=dsr > 0.85,
        verdict=verdict,
    )

    logger.info(f"DSR: {result.log_summary()}")
    return result


def deflated_sharpe_from_returns(
    returns: np.ndarray,
    n_trials: int,
    annualise: bool = True,
) -> DSRResult:
    """
    Convenience wrapper: compute DSR directly from a return series.

    Parameters
    ----------
    returns  : Array of daily strategy returns
    n_trials : Number of parameter combinations tested
    annualise: If True, annualise the Sharpe (multiply by sqrt(252))
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    t = len(returns)
    if t < 10:
        raise ValueError(f"Insufficient returns: {t} observations")

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r < 1e-10:
        raise ValueError("Zero variance in return series — cannot compute Sharpe")

    sr = mean_r / std_r
    if annualise:
        sr *= np.sqrt(252)

    # Skewness and excess kurtosis
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
    skew = float(scipy_skew(returns))
    kurt = float(scipy_kurt(returns))  # scipy returns excess kurtosis by default

    return deflated_sharpe_ratio(
        sr_observed=sr,
        t=t,
        skew=skew,
        kurt=kurt,
        n_trials=n_trials,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SURFACE ANALYSIS
# Generates DSR across a grid of N_trials to show sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def dsr_sensitivity_to_trials(
    returns: np.ndarray,
    trials_range: list[int] = None,
) -> dict[int, DSRResult]:
    """
    Compute DSR across a range of N_trials values.
    Shows how many untested variants the edge can survive before failing.

    Useful for: "I tested 50 variants but may have implicitly tried more
    through hypothesis iteration — at what N_trials does DSR drop below 0.85?"
    """
    if trials_range is None:
        trials_range = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

    results = {}
    for n in trials_range:
        try:
            results[n] = deflated_sharpe_from_returns(returns, n_trials=n)
        except Exception as e:
            logger.warning(f"DSR failed for N_trials={n}: {e}")

    # Find break-even N_trials (where DSR first drops below 0.85)
    break_even = None
    for n in sorted(results.keys()):
        if not results[n].passes_85:
            break_even = n
            break

    if break_even:
        logger.info(
            f"DSR drops below 0.85 at N_trials={break_even}. "
            f"Edge is fragile if more than {break_even} variants were implicitly tested."
        )
    else:
        logger.info(
            f"DSR remains above 0.85 across all tested N_trials values up to "
            f"{max(results.keys())}."
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EFFECTIVE N_TRIALS ESTIMATOR
# Estimates effective independent trials from correlated strategy variants
# ─────────────────────────────────────────────────────────────────────────────
def effective_n_trials(returns_matrix: np.ndarray) -> int:
    """
    Estimate effective number of independent trials from a matrix of
    correlated strategy variant returns.

    Uses eigenvalue decomposition of the correlation matrix.
    Effective N ≈ sum(λ_i)² / sum(λ_i²) where λ are eigenvalues.
    This is the participation ratio — number of "effective dimensions".

    Parameters
    ----------
    returns_matrix : (T × N) array where each column is a strategy variant

    Returns
    -------
    Effective number of independent trials (integer, ≥ 1)
    """
    T, N = returns_matrix.shape
    if N <= 1:
        return N

    # Remove columns with zero variance
    stds = np.std(returns_matrix, axis=0, ddof=1)
    valid = stds > 1e-10
    if valid.sum() < 2:
        return int(valid.sum())

    clean = returns_matrix[:, valid]
    corr = np.corrcoef(clean.T)

    # Clip correlation matrix to be positive semi-definite
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 0)

    sum_sq = float(np.sum(eigenvalues) ** 2)
    sq_sum = float(np.sum(eigenvalues ** 2))

    if sq_sum < 1e-10:
        return 1

    effective = int(np.round(sum_sq / sq_sum))
    effective = max(1, min(effective, N))

    logger.info(
        f"Effective N_trials: {effective} from {N} variants "
        f"(participation ratio method)"
    )
    return effective


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n── DSR Example (from PDF Section 5.3) ──")
    result = deflated_sharpe_ratio(
        sr_observed=0.80,
        t=756,      # 3 years daily
        skew=0.15,
        kurt=1.20,
        n_trials=50,
    )
    print(f"DSR: {result.dsr:.4f}")
    print(f"Verdict: {result.verdict}")
    print(f"SR benchmark (expected max under noise): {result.sr_benchmark:.3f}")
    print(f"SR* (corrected): {result.sr_star:.3f}")
    print(f"\nAcceptance: DSR>0.95={'✓' if result.passes_95 else '✗'} | "
          f"DSR>0.85={'✓' if result.passes_85 else '✗'}")

    print("\n── DSR Sensitivity to N_trials ──")
    np.random.seed(42)
    fake_returns = np.random.normal(0.0003, 0.01, 756)
    sensitivity = dsr_sensitivity_to_trials(fake_returns,
                                             trials_range=[1, 10, 50, 100, 500])
    for n, r in sensitivity.items():
        marker = "✓" if r.passes_85 else "✗"
        print(f"  N_trials={n:4d}: DSR={r.dsr:.4f} {marker}")
