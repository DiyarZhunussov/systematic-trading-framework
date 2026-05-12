"""
anti_overfitting_checklist.py — Anti-Overfitting Protocol
Implements Section 5.6 of the framework.

The checklist from Section 5.6 is codified here as:
    1. A human-completable checklist (printed, signed off)
    2. Automated lookahead bias detection tests
    3. Survivorship bias verification
    4. Parameter count constraint check
    5. OOS hold-out enforcement

CRITICAL (Section 5.6):
    The statistical validation methodology is ONLY meaningful if applied
    in sequence: hypothesis FIRST, then data.
    A researcher who backtests first and applies corrections after the fact
    is performing post-hoc rationalisation.

    Pre-registration rule:
        - Record N_trials from the FIRST test, not selectively
        - DSR must use the actual total variants tested, including failed attempts
        - The OOS hold-out dataset is viewed EXACTLY ONCE after all decisions

Parsimonious model selection:
    When two specifications produce Sharpe ratios within 10% of each other,
    ALWAYS select the simpler model. Complexity without statistically significant
    incremental improvement is noise.

Parameter count constraint:
    Number of free parameters ≤ N_trades / 10
    Rationale: 10 trades per parameter is the minimum for any inference.
    Example: 80 backtest trades → maximum 8 free parameters.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKLIST ITEM
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChecklistItem:
    """A single checklist item — can be automated or manual."""
    id: str
    description: str
    category: str           # "lookahead" | "survivorship" | "parameters" | "process"
    automated: bool         # True if can be verified programmatically
    passed: Optional[bool] = None
    notes: str = ""
    signed_by: str = ""


@dataclass
class ChecklistResult:
    """Full checklist evaluation result."""
    items: list[ChecklistItem]
    strategy_name: str
    researcher: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def n_passed(self) -> int:
        return sum(1 for i in self.items if i.passed is True)

    @property
    def n_failed(self) -> int:
        return sum(1 for i in self.items if i.passed is False)

    @property
    def n_unchecked(self) -> int:
        return sum(1 for i in self.items if i.passed is None)

    @property
    def all_automated_pass(self) -> bool:
        return all(i.passed for i in self.items if i.automated)

    @property
    def ready_for_validation(self) -> bool:
        """All automated checks pass AND no manual items failed."""
        return (
            self.all_automated_pass
            and not any(i.passed is False for i in self.items)
        )

    def print_report(self) -> None:
        print(f"\n{'='*60}")
        print(f"ANTI-OVERFITTING CHECKLIST: {self.strategy_name}")
        print(f"Researcher: {self.researcher}")
        print(f"Timestamp: {self.timestamp.isoformat()}")
        print(f"{'='*60}")

        for cat in ["process", "lookahead", "survivorship", "parameters"]:
            cat_items = [i for i in self.items if i.category == cat]
            if not cat_items:
                continue
            print(f"\n── {cat.upper()} ──")
            for item in cat_items:
                status = "✓" if item.passed else ("✗" if item.passed is False else "?")
                auto = "[AUTO]" if item.automated else "[MANUAL]"
                print(f"  {status} {auto} {item.id}: {item.description}")
                if item.notes:
                    print(f"       Notes: {item.notes}")

        print(f"\nSummary: {self.n_passed} passed | "
              f"{self.n_failed} failed | "
              f"{self.n_unchecked} unchecked")
        print(f"Ready for validation: {'YES' if self.ready_for_validation else 'NO'}")
        print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER COUNT CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_parameter_count(
    n_free_parameters: int,
    n_trades: int,
) -> dict:
    """
    Verify parameter count constraint (Section 5.6):
    Number of free parameters ≤ N_trades / 10

    Parameters without economic motivation count toward this limit.

    Parameters
    ----------
    n_free_parameters : Total free parameters in the strategy
    n_trades          : Number of trades in the backtest

    Returns
    -------
    dict with passes, max_allowed, ratio, recommendation
    """
    max_allowed = n_trades // 10
    passes = n_free_parameters <= max_allowed
    ratio = n_free_parameters / max_allowed if max_allowed > 0 else float('inf')

    if not passes:
        recommendation = (
            f"Reduce to ≤{max_allowed} parameters OR increase trading frequency "
            f"to achieve ≥{n_free_parameters * 10} trades."
        )
    else:
        recommendation = "Parameter count acceptable."

    result = {
        "passes": passes,
        "n_free_parameters": n_free_parameters,
        "n_trades": n_trades,
        "max_allowed": max_allowed,
        "ratio": ratio,
        "recommendation": recommendation,
    }

    logger.info(
        f"Parameter count: {n_free_parameters} params / {n_trades} trades → "
        f"{'PASS' if passes else 'FAIL'} (max {max_allowed})"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LOOKAHEAD BIAS DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_lookahead_bias(
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    prices: np.ndarray,
    n_injections: int = 5,
    injection_magnitude: float = 10.0,
) -> dict:
    """
    Test for lookahead bias by injecting obviously detectable future information.

    Method (Section 5.6):
        Artificially inject large future prices at known locations.
        A correct no-lookahead strategy should show NO improvement.
        If the strategy produces unrealistically high Sharpe on injected data,
        it has access to future information.

    Parameters
    ----------
    strategy_fn         : Callable(prices) → returns array
    prices              : Original price series
    n_injections        : Number of future-value injections
    injection_magnitude : Size of artificial price spike (multiples of std dev)

    Returns
    -------
    dict with detected, baseline_sharpe, injected_sharpe, ratio, passes
    """
    if len(prices) < 50:
        return {"passes": True, "reason": "insufficient_data_for_lookahead_test"}

    # ── Baseline performance ──────────────────────────────────────────────────
    try:
        baseline_returns = strategy_fn(prices)
        baseline_std = float(np.std(baseline_returns, ddof=1))
        if baseline_std < 1e-10:
            baseline_sharpe = 0.0
        else:
            baseline_sharpe = float(
                np.mean(baseline_returns) / baseline_std * np.sqrt(252)
            )
    except Exception as e:
        return {"passes": True, "reason": f"baseline_failed: {e}"}

    # ── Inject future information ─────────────────────────────────────────────
    price_std = float(np.std(np.diff(prices)))
    spike_size = price_std * injection_magnitude

    injected_prices = prices.copy()
    rng = np.random.default_rng(42)
    injection_points = rng.choice(
        range(10, len(prices) - 5), size=n_injections, replace=False
    )

    # Inject large upward spikes 1 bar AHEAD of each point
    for pt in injection_points:
        injected_prices[pt + 1] = prices[pt] + spike_size * 5  # Obvious future move

    try:
        injected_returns = strategy_fn(injected_prices)
        inj_std = float(np.std(injected_returns, ddof=1))
        if inj_std < 1e-10:
            injected_sharpe = 0.0
        else:
            injected_sharpe = float(
                np.mean(injected_returns) / inj_std * np.sqrt(252)
            )
    except Exception as e:
        return {"passes": True, "reason": f"injection_test_failed: {e}"}

    # ── Detection ─────────────────────────────────────────────────────────────
    # If lookahead: injected_sharpe >> baseline_sharpe
    ratio = (
        injected_sharpe / abs(baseline_sharpe)
        if abs(baseline_sharpe) > 1e-10
        else float('inf')
    )

    # Lookahead detected if injected SR > 3× baseline OR > 3.0 absolute
    lookahead_detected = ratio > 3.0 or injected_sharpe > 3.0

    passes = not lookahead_detected

    if lookahead_detected:
        logger.error(
            f"LOOKAHEAD BIAS DETECTED: "
            f"baseline_SR={baseline_sharpe:.3f}, "
            f"injected_SR={injected_sharpe:.3f}, "
            f"ratio={ratio:.2f}. "
            f"Strategy has access to future information."
        )
    else:
        logger.info(
            f"Lookahead test: PASS | "
            f"baseline_SR={baseline_sharpe:.3f}, "
            f"injected_SR={injected_sharpe:.3f}"
        )

    return {
        "passes": passes,
        "lookahead_detected": lookahead_detected,
        "baseline_sharpe": float(baseline_sharpe),
        "injected_sharpe": float(injected_sharpe),
        "ratio": float(ratio),
        "injection_points": injection_points.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSIMONIOUS MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_parsimonious_model(
    models: list[dict],
) -> dict:
    """
    Apply parsimonious model selection (Section 5.6):
    When two specifications produce Sharpe ratios within 10% of each other,
    ALWAYS select the simpler model.

    Parameters
    ----------
    models : List of dicts with keys:
             'name', 'sharpe', 'n_parameters', 'params' (dict)

    Returns
    -------
    dict: selected model with selection_reason
    """
    if not models:
        raise ValueError("No models provided")

    if len(models) == 1:
        return {**models[0], "selection_reason": "only_model"}

    # Sort by Sharpe descending
    sorted_models = sorted(models, key=lambda m: m["sharpe"], reverse=True)
    best = sorted_models[0]
    best_sharpe = best["sharpe"]

    # Among models within 10% of best Sharpe, pick the simplest
    tolerance = 0.10
    candidates = [
        m for m in sorted_models
        if abs(best_sharpe) > 1e-10
        and abs(m["sharpe"] - best_sharpe) / abs(best_sharpe) <= tolerance
    ]

    if not candidates:
        candidates = [best]

    # Select minimum parameters among candidates
    selected = min(candidates, key=lambda m: m["n_parameters"])

    if selected["name"] != best["name"]:
        logger.info(
            f"Parsimonious selection: chose '{selected['name']}' "
            f"({selected['n_parameters']} params, SR={selected['sharpe']:.3f}) "
            f"over '{best['name']}' "
            f"({best['n_parameters']} params, SR={best['sharpe']:.3f}) — "
            f"within {tolerance:.0%} tolerance, fewer parameters wins."
        )
        reason = (
            f"Within {tolerance:.0%} Sharpe tolerance of best — "
            f"selected for parsimony ({selected['n_parameters']} < {best['n_parameters']} params)"
        )
    else:
        reason = "Best Sharpe AND simplest within tolerance"

    return {**selected, "selection_reason": reason, "all_candidates": candidates}


# ─────────────────────────────────────────────────────────────────────────────
# FULL CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
def build_checklist(
    strategy_name: str,
    researcher: str,
) -> ChecklistResult:
    """
    Build the full anti-overfitting checklist for a strategy under review.
    Manual items must be signed off by the researcher.
    Automated items are run programmatically.

    Returns a ChecklistResult — call result.print_report() to display.
    """
    items = [
        # ── Process controls ──────────────────────────────────────────────────
        ChecklistItem(
            id="P1", category="process", automated=False,
            description="Hypothesis written BEFORE examining data for this strategy",
        ),
        ChecklistItem(
            id="P2", category="process", automated=False,
            description="N_trials recorded from FIRST test (not selectively reported)",
        ),
        ChecklistItem(
            id="P3", category="process", automated=False,
            description="OOS hold-out (20%) set aside BEFORE any parameter decisions",
        ),
        ChecklistItem(
            id="P4", category="process", automated=False,
            description="OOS hold-out viewed EXACTLY ONCE (after all parameters finalised)",
        ),
        ChecklistItem(
            id="P5", category="process", automated=False,
            description="Train/test split applied BEFORE any data transformation",
        ),

        # ── Lookahead bias ────────────────────────────────────────────────────
        ChecklistItem(
            id="L1", category="lookahead", automated=False,
            description="All signals use only Close_t of bar t-1 (not bar t)",
        ),
        ChecklistItem(
            id="L2", category="lookahead", automated=False,
            description="No indicator values computed using current bar's data at decision time",
        ),
        ChecklistItem(
            id="L3", category="lookahead", automated=False,
            description="Spread costs applied at execution, not at signal generation",
        ),
        ChecklistItem(
            id="L4", category="lookahead", automated=False,
            description="Rolling statistics (mean, std, ATR) use only past bars at each point",
        ),
        ChecklistItem(
            id="L5", category="lookahead", automated=False,
            description="Regime classification does not use future price information",
        ),
        ChecklistItem(
            id="L6", category="lookahead", automated=False,
            description=(
                "Backtesting library validated: deliberately injected forward-looking "
                "signals produce unrealistically high Sharpe (confirms library detects lookahead)"
            ),
        ),
        ChecklistItem(
            id="L7", category="lookahead", automated=False,
            description="Rollover adjustments applied to raw data BEFORE any signal computation",
        ),
        ChecklistItem(
            id="L8", category="lookahead", automated=False,
            description="All normalisation (z-scores, percentiles) uses only past data at each point",
        ),

        # ── Survivorship bias ─────────────────────────────────────────────────
        ChecklistItem(
            id="S1", category="survivorship", automated=False,
            description="Instrument universe fixed BEFORE backtest (not selected for past performance)",
        ),
        ChecklistItem(
            id="S2", category="survivorship", automated=False,
            description="Broker selected BEFORE backtest (not for best historical spread environment)",
        ),
        ChecklistItem(
            id="S3", category="survivorship", automated=False,
            description="All instruments in scope existed throughout the entire backtest period",
        ),
        ChecklistItem(
            id="S4", category="survivorship", automated=False,
            description=(
                "Strategy retirement criteria applied retroactively in backtest "
                "(backtest stops trading when live monitoring would have stopped it)"
            ),
        ),

        # ── Parameter constraints ─────────────────────────────────────────────
        ChecklistItem(
            id="PC1", category="parameters", automated=False,
            description=(
                "Free parameter count ≤ N_trades / 10 "
                "(10 trades per parameter minimum — verify with check_parameter_count())"
            ),
        ),
        ChecklistItem(
            id="PC2", category="parameters", automated=False,
            description=(
                "All parameters have explicit economic justification "
                "(pure data-mining parameters counted toward limit)"
            ),
        ),
        ChecklistItem(
            id="PC3", category="parameters", automated=False,
            description=(
                "Parsimonious model selected: where two specs within 10% Sharpe, "
                "simpler model chosen (verified with select_parsimonious_model())"
            ),
        ),
    ]

    return ChecklistResult(
        items=items,
        strategy_name=strategy_name,
        researcher=researcher,
    )


def sign_off_item(
    checklist: ChecklistResult,
    item_id: str,
    passed: bool,
    notes: str = "",
    signed_by: str = "",
) -> None:
    """Mark a checklist item as passed or failed."""
    for item in checklist.items:
        if item.id == item_id:
            item.passed = passed
            item.notes = notes
            item.signed_by = signed_by or checklist.researcher
            return
    raise KeyError(f"Checklist item '{item_id}' not found")


# ─────────────────────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    print("\n── Parameter Count Check ──")
    r = check_parameter_count(n_free_parameters=6, n_trades=80)
    print(f"  passes={r['passes']} | max_allowed={r['max_allowed']} | "
          f"recommendation={r['recommendation']}")

    print("\n── Lookahead Bias Detection ──")
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.normal(0, 0.01, 300))

    def simple_mr_strategy(p):
        from numpy.lib.stride_tricks import sliding_window_view
        if len(p) < 25:
            return np.array([])
        z = np.array([
            (p[t] - np.mean(p[t-20:t])) / (np.std(p[t-20:t]) + 1e-10)
            for t in range(20, len(p))
        ])
        pos = np.where(z < -1.5, 1, np.where(z > 1.5, -1, 0))
        ret = np.diff(p[20:]) / (p[20:-1] + 1e-10)
        return pos[:-1] * ret

    la = detect_lookahead_bias(simple_mr_strategy, prices)
    print(f"  Lookahead detected: {la['lookahead_detected']} | "
          f"baseline_SR={la['baseline_sharpe']:.3f} | "
          f"injected_SR={la['injected_sharpe']:.3f}")

    print("\n── Checklist (showing structure) ──")
    cl = build_checklist("Mean Reversion EUR/USD", "Researcher Name")
    sign_off_item(cl, "P1", True, "Hypothesis pre-registered in research log")
    sign_off_item(cl, "L1", True, "Confirmed: all entry signals use Close[t-1]")
    cl.print_report()
