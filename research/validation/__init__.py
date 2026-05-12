# research/validation/__init__.py
from .deflated_sharpe import deflated_sharpe_ratio, deflated_sharpe_from_returns, DSRResult
from .walk_forward import (
    run_walk_forward, generate_walk_forward_folds, final_holdout_validation,
    WalkForwardResult, WalkForwardFold, FoldResult,
)
from .pbo import (
    probability_of_backtest_overfitting, build_strategy_returns_matrix,
    PBOResult,
)
from .multiple_testing import (
    benjamini_hochberg, bonferroni_correction,
    apply_framework_significance_rule, sharpe_to_pvalue,
    BHResult, BonferroniResult,
)
from .anti_overfitting_checklist import (
    build_checklist, sign_off_item, check_parameter_count,
    detect_lookahead_bias, select_parsimonious_model,
    ChecklistResult, ChecklistItem,
)
from .opportunity_ranker import (
    OpportunityRanker, OpportunityScore,
    get_framework_scores, RESEARCH_THRESHOLD,
)
