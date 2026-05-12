"""
opportunity_ranker.py — Research Opportunity Ranking Scorecard
Implements Section 10.2 of the framework.

Purpose:
    Before allocating research resources to any hypothesis, score it on
    five dimensions. Research investment is warranted ONLY for total
    scores above 0.60.

Scoring dimensions and criteria (Section 10.2):

    1. Edge Persistence
       Score 5: Behavioural mechanism (slow to arbitrage)
       Score 3: Structural (can change with regulations)
       Score 1: Pure statistical pattern (likely noise)

    2. Implementation Simplicity
       Score 5: Few parameters, interpretable
       Score 3: Moderate complexity, justified
       Score 1: High complexity, many parameters

    3. Robustness
       Score 5: Similar performance across instruments and periods
       Score 3: Works in most regimes, known failures
       Score 1: Narrow conditions only

    4. Execution Feasibility
       Score 5: Works at retail fill speeds and costs
       Score 3: Marginal, careful execution needed
       Score 1: Requires HFT infrastructure

    5. Operational Complexity
       Score 5: Easy to monitor and diagnose
       Score 3: Moderate monitoring burden
       Score 1: Complex failure modes

Current rankings from framework (Section 10.2):
    Trend Following     : 0.88
    Regime Adaptation   : 0.76
    Mean Reversion      : 0.68
    Vol Breakout        : 0.60
    Statistical Arbitrage: 0.44 ← Does NOT meet threshold
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum score to warrant research investment (Section 10.2)
RESEARCH_THRESHOLD = 0.60

# Maximum possible score per dimension
MAX_SCORE_PER_DIM = 5

# Number of dimensions
N_DIMENSIONS = 5


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DESCRIPTORS (from Section 10.2 table)
# ─────────────────────────────────────────────────────────────────────────────
DIMENSION_DESCRIPTORS = {
    "edge_persistence": {
        5: "Behavioural mechanism (slow to arbitrage — e.g. underreaction, herding)",
        3: "Structural (can change with regulations or market structure)",
        1: "Pure statistical pattern (likely noise — no economic rationale)",
    },
    "implementation_simplicity": {
        5: "Few parameters, fully interpretable, easy to explain",
        3: "Moderate complexity with justified parameters",
        1: "High complexity, many parameters, difficult to diagnose",
    },
    "robustness": {
        5: "Similar performance across multiple instruments and time periods",
        3: "Works in most regimes with known, documented failure modes",
        1: "Narrow conditions only — regime-specific or instrument-specific",
    },
    "execution_feasibility": {
        5: "Works at retail fill speeds and effective costs",
        3: "Marginal — requires careful execution and tight spread management",
        1: "Requires HFT infrastructure (sub-millisecond co-location)",
    },
    "operational_complexity": {
        5: "Easy to monitor and diagnose — clear failure signals",
        3: "Moderate monitoring burden — requires dedicated attention",
        1: "Complex failure modes — hard to distinguish alpha decay from bad luck",
    },
}

# Normalisation: score = raw / max_raw
# max_raw = 5 × N_DIMENSIONS = 25, normalised = sum/25
def _normalise(raw_total: float) -> float:
    return raw_total / (MAX_SCORE_PER_DIM * N_DIMENSIONS)


# ─────────────────────────────────────────────────────────────────────────────
# OPPORTUNITY HYPOTHESIS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OpportunityScore:
    """Scored research opportunity hypothesis."""
    name: str
    description: str
    researcher: str

    # Dimension scores (1, 3, or 5 per dimension)
    edge_persistence: int
    implementation_simplicity: int
    robustness: int
    execution_feasibility: int
    operational_complexity: int

    # Qualitative notes per dimension
    edge_notes: str = ""
    simplicity_notes: str = ""
    robustness_notes: str = ""
    execution_notes: str = ""
    operations_notes: str = ""

    # Economic rationale (required for Stage 1 gate — Section 10.1)
    economic_rationale: str = ""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        for dim, val in [
            ("edge_persistence", self.edge_persistence),
            ("implementation_simplicity", self.implementation_simplicity),
            ("robustness", self.robustness),
            ("execution_feasibility", self.execution_feasibility),
            ("operational_complexity", self.operational_complexity),
        ]:
            if val not in (1, 3, 5):
                raise ValueError(
                    f"Score for '{dim}' must be 1, 3, or 5 (got {val})"
                )

    @property
    def raw_total(self) -> int:
        return (
            self.edge_persistence
            + self.implementation_simplicity
            + self.robustness
            + self.execution_feasibility
            + self.operational_complexity
        )

    @property
    def normalised_score(self) -> float:
        return _normalise(self.raw_total)

    @property
    def warrants_research(self) -> bool:
        return self.normalised_score >= RESEARCH_THRESHOLD

    @property
    def has_economic_rationale(self) -> bool:
        return len(self.economic_rationale.strip()) > 20

    def stage1_gate_passes(self) -> tuple[bool, str]:
        """
        Stage 1 gate: Economic plausibility (Section 10.1).
        Gate is on economic rationale, NOT statistical performance.
        """
        if not self.warrants_research:
            return (
                False,
                f"Score {self.normalised_score:.2f} < threshold {RESEARCH_THRESHOLD}. "
                f"Do not invest research resources."
            )
        if not self.has_economic_rationale:
            return (
                False,
                "Economic rationale not provided or too brief. "
                "Stage 1 gate requires written economic mechanism."
            )
        return True, "Stage 1 gate passed — proceed to initial research"

    def print_scorecard(self) -> None:
        print(f"\n{'='*60}")
        print(f"RESEARCH OPPORTUNITY SCORECARD: {self.name}")
        print(f"Researcher: {self.researcher}")
        print(f"{'='*60}")
        print(f"\nDescription: {self.description}")
        print(f"\nEconomic Rationale: {self.economic_rationale}")
        print(f"\nScores:")
        print(f"  Edge Persistence        : {self.edge_persistence}/5  — {DIMENSION_DESCRIPTORS['edge_persistence'].get(self.edge_persistence, '')}")
        if self.edge_notes:
            print(f"    Notes: {self.edge_notes}")
        print(f"  Implementation Simplicity: {self.implementation_simplicity}/5  — {DIMENSION_DESCRIPTORS['implementation_simplicity'].get(self.implementation_simplicity, '')}")
        if self.simplicity_notes:
            print(f"    Notes: {self.simplicity_notes}")
        print(f"  Robustness              : {self.robustness}/5  — {DIMENSION_DESCRIPTORS['robustness'].get(self.robustness, '')}")
        if self.robustness_notes:
            print(f"    Notes: {self.robustness_notes}")
        print(f"  Execution Feasibility   : {self.execution_feasibility}/5  — {DIMENSION_DESCRIPTORS['execution_feasibility'].get(self.execution_feasibility, '')}")
        if self.execution_notes:
            print(f"    Notes: {self.execution_notes}")
        print(f"  Operational Complexity  : {self.operational_complexity}/5  — {DIMENSION_DESCRIPTORS['operational_complexity'].get(self.operational_complexity, '')}")
        if self.operations_notes:
            print(f"    Notes: {self.operations_notes}")
        print(f"\nTotal: {self.raw_total}/{MAX_SCORE_PER_DIM * N_DIMENSIONS} "
              f"= {self.normalised_score:.2f}")
        passes, reason = self.stage1_gate_passes()
        print(f"Stage 1 Gate: {'PASS ✓' if passes else 'FAIL ✗'} — {reason}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PRE-SCORED FRAMEWORK STRATEGIES (Section 10.2 table)
# ─────────────────────────────────────────────────────────────────────────────
def get_framework_scores() -> list[OpportunityScore]:
    """
    Return the pre-scored strategies from Section 10.2.
    These are reference benchmarks for comparing new hypotheses.
    """
    return [
        OpportunityScore(
            name="Trend Following",
            description="Dual EMA crossover + Donchian channel on FX and indices",
            researcher="Framework v2.0",
            edge_persistence=5,
            implementation_simplicity=4,
            robustness=5,
            execution_feasibility=4,
            operational_complexity=4,
            economic_rationale=(
                "Investor underreaction creates autocorrelation over 1–12 months. "
                "Central banks slow but don't reverse trends. Interest rate divergence "
                "creates multi-week FX themes. Crisis alpha: positive returns during "
                "equity crashes because no equity exposure (Kim, Tse & Wald 2016)."
            ),
        ),
        OpportunityScore(
            name="Regime Adaptation",
            description="ADX-based regime detection modulating strategy allocations",
            researcher="Framework v2.0",
            edge_persistence=4,
            implementation_simplicity=3,
            robustness=4,
            execution_feasibility=5,
            operational_complexity=3,
            economic_rationale=(
                "Different market regimes systematically favour different strategies. "
                "Choppy regimes favour mean reversion; trending regimes favour momentum. "
                "Detecting and adapting to regime reduces loss during hostile periods."
            ),
        ),
        OpportunityScore(
            name="Intraday Mean Reversion",
            description="Z-score reversion on 5-minute bars with autocorrelation filter",
            researcher="Framework v2.0",
            edge_persistence=3,
            implementation_simplicity=4,
            robustness=3,
            execution_feasibility=3,
            operational_complexity=4,
            economic_rationale=(
                "Short-term liquidity imbalances create temporary price dislocations. "
                "As institutional flow exhausts and market makers rebalance, prices "
                "revert. Edge is compensation for bearing short-term inventory risk."
            ),
        ),
        OpportunityScore(
            name="Volatility Breakout",
            description="ATR ratio + Bollinger squeeze → breakout entry on 4H bars",
            researcher="Framework v2.0",
            edge_persistence=3,
            implementation_simplicity=3,
            robustness=3,
            execution_feasibility=3,
            operational_complexity=3,
            economic_rationale=(
                "Volatility is mean-reverting at medium timescales. Extreme compression "
                "precedes expansion. Institutional flow constrained during thin markets "
                "enters when vol expands, amplifying initial moves."
            ),
        ),
        OpportunityScore(
            name="Statistical Arbitrage",
            description="Cointegrated pairs (EUR/USD–GBP/USD, NQ–SPX, XAU–JPY)",
            researcher="Framework v2.0",
            edge_persistence=3,
            implementation_simplicity=2,
            robustness=2,
            execution_feasibility=2,
            operational_complexity=2,
            economic_rationale=(
                "Co-integrated instruments share common fundamental drivers. Temporary "
                "divergence creates mean-reversion opportunity. However: leg execution "
                "risk at MT5 speeds represents 20–30% of gross edge. Not viable until "
                "simultaneous execution is available."
            ),
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RANKER
# ─────────────────────────────────────────────────────────────────────────────
class OpportunityRanker:
    """
    Maintains a ranked list of research opportunity hypotheses.
    Provides comparison against framework benchmarks and allocation guidance.
    """

    def __init__(self):
        self._opportunities: list[OpportunityScore] = []

    def add(self, opportunity: OpportunityScore) -> None:
        """Add a new research opportunity to the ranker."""
        passes, reason = opportunity.stage1_gate_passes()
        if not passes:
            logger.warning(
                f"Opportunity '{opportunity.name}' does not pass Stage 1 gate: {reason}"
            )
        else:
            logger.info(
                f"Opportunity '{opportunity.name}' added: "
                f"score={opportunity.normalised_score:.2f}"
            )
        self._opportunities.append(opportunity)

    def ranked(self) -> list[OpportunityScore]:
        """Return opportunities sorted by normalised score descending."""
        return sorted(
            self._opportunities,
            key=lambda o: o.normalised_score,
            reverse=True,
        )

    def eligible_for_research(self) -> list[OpportunityScore]:
        """Return only opportunities that meet the research threshold."""
        return [o for o in self.ranked() if o.warrants_research]

    def print_ranking(self) -> None:
        """Print full ranked scorecard."""
        print(f"\n{'='*70}")
        print(f"RESEARCH OPPORTUNITY RANKINGS")
        print(f"Threshold for research investment: {RESEARCH_THRESHOLD}")
        print(f"{'='*70}")
        print(f"{'Rank':<5} {'Name':<30} {'Score':<8} {'Gate':<8} {'Persst':<8} {'Simpl':<8} {'Robus':<8} {'Exec':<8} {'Ops':<8}")
        print(f"{'-'*70}")
        for i, opp in enumerate(self.ranked(), 1):
            passes, _ = opp.stage1_gate_passes()
            gate_str = "PASS" if passes else "FAIL"
            marker = "✓" if opp.warrants_research else "✗"
            print(
                f"{i:<5} {opp.name:<30} "
                f"{marker}{opp.normalised_score:.2f}   "
                f"{gate_str:<8} "
                f"{opp.edge_persistence:<8} "
                f"{opp.implementation_simplicity:<8} "
                f"{opp.robustness:<8} "
                f"{opp.execution_feasibility:<8} "
                f"{opp.operational_complexity:<8}"
            )
        print(f"{'='*70}")
        eligible = len(self.eligible_for_research())
        print(
            f"\n{eligible}/{len(self._opportunities)} opportunities "
            f"eligible for research investment (score ≥ {RESEARCH_THRESHOLD})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n── Framework Strategy Rankings (Section 10.2) ──")
    ranker = OpportunityRanker()
    for opp in get_framework_scores():
        ranker.add(opp)

    ranker.print_ranking()

    print("\n── Statistical Arbitrage Detail ──")
    stat_arb = [o for o in ranker.ranked() if o.name == "Statistical Arbitrage"][0]
    stat_arb.print_scorecard()
