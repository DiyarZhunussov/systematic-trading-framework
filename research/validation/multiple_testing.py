"""
multiple_testing.py — Multiple Testing Correction
Implements Section 5.5 of the framework.

The problem (Section 5.5):
    Testing 50 strategy variants at α=5% significance produces an expected
    2.5 false discoveries even if NONE of the strategies have any edge.

Two methods:

1. Bonferroni correction (conservative):
    Adjusted α = α_target / N_tests
    Example: 50 tests, α_target=0.05 → adjusted α = 0.001
    Use when: tests are independent or highly correlated
    Controls: Family-Wise Error Rate (FWER)

2. Benjamini-Hochberg (BH) procedure (recommended for correlated tests):
    1. Rank all p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(N)
    2. Find largest k: p_(k) ≤ (k/N) × FDR_target
    3. Reject null for tests 1 through k
    Controls: False Discovery Rate (FDR)
    More powerful than Bonferroni for correlated strategies (typical in trading)
    FDR_target recommended: 0.05

Operational rule (Section 5.5):
    No strategy is allocated capital unless it achieves p < 0.01 on its primary
    hypothesis test, WITH Benjamini-Hochberg correction across all tested variants.
    The 0.01 threshold (not 0.05) is required because the 2026 FDR evidence
    indicates even conservative procedures face substantial false discovery rates.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BonferroniResult:
    """Result of Bonferroni correction."""
    n_tests: int
    alpha_target: float
    adjusted_alpha: float
    p_values: list[float]
    rejected: list[bool]        # True = reject null (significant)
    n_rejected: int
    strategy_names: list[str]

    def significant_strategies(self) -> list[str]:
        return [
            self.strategy_names[i]
            for i, r in enumerate(self.rejected) if r
        ]

    def log_summary(self) -> str:
        return (
            f"Bonferroni | N={self.n_tests} | "
            f"adj_α={self.adjusted_alpha:.4f} | "
            f"rejected={self.n_rejected}/{self.n_tests}"
        )


@dataclass
class BHResult:
    """Result of Benjamini-Hochberg procedure."""
    n_tests: int
    fdr_target: float
    p_values: list[float]
    sorted_p_values: list[float]
    bh_thresholds: list[float]   # (k/N) × FDR_target for each rank k
    rejected: list[bool]         # True = reject null
    n_rejected: int
    strategy_names: list[str]
    critical_k: int              # Largest k where p_(k) ≤ threshold

    def significant_strategies(self) -> list[str]:
        return [
            self.strategy_names[i]
            for i, r in enumerate(self.rejected) if r
        ]

    def log_summary(self) -> str:
        return (
            f"Benjamini-Hochberg | N={self.n_tests} | "
            f"FDR_target={self.fdr_target:.2f} | "
            f"rejected={self.n_rejected}/{self.n_tests} | "
            f"critical_k={self.critical_k}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BONFERRONI CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
def bonferroni_correction(
    p_values: list[float],
    strategy_names: list[str] = None,
    alpha_target: float = 0.05,
) -> BonferroniResult:
    """
    Apply Bonferroni correction to a set of p-values.

    Controls the Family-Wise Error Rate (FWER).
    Conservative — appropriate when tests are independent.
    For correlated strategy variants, use benjamini_hochberg() instead.

    Parameters
    ----------
    p_values       : List of p-values, one per strategy variant
    strategy_names : Optional names for each strategy
    alpha_target   : Target significance level (default 0.05)

    Returns
    -------
    BonferroniResult
    """
    n = len(p_values)
    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n)]

    adjusted_alpha = alpha_target / n
    rejected = [p <= adjusted_alpha for p in p_values]
    n_rejected = sum(rejected)

    result = BonferroniResult(
        n_tests=n,
        alpha_target=alpha_target,
        adjusted_alpha=adjusted_alpha,
        p_values=list(p_values),
        rejected=rejected,
        n_rejected=n_rejected,
        strategy_names=strategy_names,
    )

    logger.info(f"Bonferroni: {result.log_summary()}")
    if n_rejected > 0:
        logger.info(f"Significant: {result.significant_strategies()}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BENJAMINI-HOCHBERG PROCEDURE
# ─────────────────────────────────────────────────────────────────────────────
def benjamini_hochberg(
    p_values: list[float],
    strategy_names: list[str] = None,
    fdr_target: float = 0.05,
) -> BHResult:
    """
    Apply Benjamini-Hochberg procedure to control False Discovery Rate.

    More powerful than Bonferroni for correlated strategies (typical in trading
    research where variants share the same underlying data and logic).

    Procedure:
        1. Rank all p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(N)
        2. Find largest k: p_(k) ≤ (k/N) × FDR_target
        3. Reject null for all tests with rank ≤ k

    Parameters
    ----------
    p_values     : List of p-values, one per strategy variant
    strategy_names : Optional names for each strategy
    fdr_target   : Target FDR (default 0.05 per Section 5.5)

    Returns
    -------
    BHResult with rejected flags and diagnostic data
    """
    n = len(p_values)
    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n)]

    # Sort p-values with original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices = [i for i, _ in indexed_p]
    sorted_p = [p for _, p in indexed_p]

    # BH thresholds: (k/N) × FDR_target for rank k (1-indexed)
    bh_thresholds = [(k + 1) / n * fdr_target for k in range(n)]

    # Find largest k where p_(k) ≤ threshold
    critical_k = -1
    for k in range(n - 1, -1, -1):
        if sorted_p[k] <= bh_thresholds[k]:
            critical_k = k
            break

    # Reject all tests with rank ≤ critical_k
    rejected_sorted = [k <= critical_k for k in range(n)]

    # Map back to original ordering
    rejected = [False] * n
    for rank, orig_idx in enumerate(sorted_indices):
        rejected[orig_idx] = rejected_sorted[rank]

    n_rejected = sum(rejected)

    result = BHResult(
        n_tests=n,
        fdr_target=fdr_target,
        p_values=list(p_values),
        sorted_p_values=sorted_p,
        bh_thresholds=bh_thresholds,
        rejected=rejected,
        n_rejected=n_rejected,
        strategy_names=strategy_names,
        critical_k=critical_k,
    )

    logger.info(f"BH: {result.log_summary()}")
    if n_rejected > 0:
        logger.info(f"Significant: {result.significant_strategies()}")
    else:
        logger.info("BH: No strategies passed FDR correction")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FRAMEWORK OPERATIONAL RULE
# The 0.01 threshold check (not 0.05) as per Section 5.5
# ─────────────────────────────────────────────────────────────────────────────
def apply_framework_significance_rule(
    p_values: list[float],
    strategy_names: list[str] = None,
) -> dict:
    """
    Apply the framework's operational significance rule (Section 5.5):

    No strategy allocated capital unless:
        1. p < 0.01 on its primary hypothesis test (not 0.05)
        2. Benjamini-Hochberg correction applied across all variants
        3. Both conditions must be satisfied

    The 0.01 threshold is required because 2026 FDR evidence shows
    even conservative procedures face substantial false discovery rates.

    Parameters
    ----------
    p_values       : Raw p-values from primary hypothesis tests
    strategy_names : Names for logging

    Returns
    -------
    dict with 'capital_eligible' list and full diagnostic breakdown
    """
    n = len(p_values)
    if strategy_names is None:
        strategy_names = [f"strategy_{i}" for i in range(n)]

    # Condition 1: p < 0.01 (strict threshold)
    passes_strict = [p < 0.01 for p in p_values]

    # Condition 2: BH correction at FDR=0.05
    bh_result = benjamini_hochberg(p_values, strategy_names, fdr_target=0.05)

    # Both conditions must be met
    capital_eligible = [
        name for i, name in enumerate(strategy_names)
        if passes_strict[i] and bh_result.rejected[i]
    ]

    # Also apply Bonferroni for comparison
    bonf_result = bonferroni_correction(p_values, strategy_names, alpha_target=0.05)

    result = {
        "n_tested": n,
        "capital_eligible": capital_eligible,
        "n_eligible": len(capital_eligible),
        "passes_strict_001": sum(passes_strict),
        "passes_bh_fdr05": bh_result.n_rejected,
        "passes_bonferroni": bonf_result.n_rejected,
        "bh_result": bh_result,
        "bonferroni_result": bonf_result,
        "framework_note": (
            "Capital allocated only where BOTH p<0.01 AND BH-corrected. "
            "0.01 threshold required per 2026 FDR evidence (Section 5.5)."
        ),
    }

    logger.info(
        f"Framework significance rule: "
        f"{len(capital_eligible)}/{n} strategies capital-eligible"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# P-VALUE COMPUTATION FROM SHARPE
# ─────────────────────────────────────────────────────────────────────────────
def sharpe_to_pvalue(
    sharpe: float,
    n_observations: int,
    two_sided: bool = True,
) -> float:
    """
    Convert an observed Sharpe ratio to a p-value under the null H₀: SR=0.
    Uses t-distribution with T-1 degrees of freedom.

    For non-normal returns this is approximate — use DSR for more accuracy.
    """
    from scipy.stats import t as t_dist
    if n_observations < 3:
        return 1.0

    # t-statistic = SR * sqrt(T)
    t_stat = sharpe * np.sqrt(n_observations)
    df = n_observations - 1

    if two_sided:
        p = float(2 * t_dist.sf(abs(t_stat), df))
    else:
        p = float(t_dist.sf(t_stat, df))

    return float(np.clip(p, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n── Multiple Testing Example: 20 strategies ──")
    np.random.seed(42)

    # Simulate p-values: 18 noise strategies, 2 genuine
    p_vals = list(np.random.uniform(0.05, 1.0, 18)) + [0.002, 0.008]
    names = [f"variant_{i}" for i in range(18)] + ["genuine_1", "genuine_2"]

    print("\nBonferroni:")
    bonf = bonferroni_correction(p_vals, names)
    print(f"  Significant: {bonf.significant_strategies()}")

    print("\nBenjamini-Hochberg (FDR=0.05):")
    bh = benjamini_hochberg(p_vals, names)
    print(f"  Significant: {bh.significant_strategies()}")

    print("\nFramework operational rule (p<0.01 AND BH-corrected):")
    rule = apply_framework_significance_rule(p_vals, names)
    print(f"  Capital eligible: {rule['capital_eligible']}")
