"""
production/engines/__init__.py
Public interface for all alpha engines.

Import pattern:
    from production.engines import (
        MeanReversionEngine, TrendFollowingEngine,
        VolatilityBreakoutEngine, RegimeEngine,
        StatisticalArbitrageEngine,
        SignalDirection, RegimeState, StrategyScales,
        get_regime_scales, normalize_strategy_scales,
    )
"""

# ── Signal direction (shared enum — use this, not engine-local versions) ─────
from production.engines.signal_engine_mean_reversion import SignalDirection

# ── Alpha engines ─────────────────────────────────────────────────────────────
from production.engines.signal_engine_mean_reversion import (
    MeanReversionEngine,
    MeanReversionSignal,
    rolling_zscore,
    rolling_autocorrelation,
    compute_atr,
    session_volatility_profile,
)

from production.engines.signal_engine_trend_following import (
    TrendFollowingEngine,
    TrendSignal,
    exponential_moving_average,
    donchian_channel,
    momentum_12m,
)

from production.engines.signal_engine_volatility_breakout import (
    VolatilityBreakoutEngine,
    BreakoutSignal,
    atr_volatility_ratio,
    bollinger_band_width,
)

from production.engines.regime_engine import (
    RegimeEngine,
    RegimeState,
    StrategyScales,
    VolatilityLevel,
    TrendState,
    get_regime_scales,
    normalize_strategy_scales,
    cusum_structural_break,
    ALLOCATION_MATRIX,
)

from production.engines.signal_engine_stat_arb import (
    StatisticalArbitrageEngine,
    StatArbSignal,
    CointegrationResult,
    engle_granger_cointegration,
    STAT_ARB_ALLOCATION_PCT,
    APPROVED_PAIRS,
)

from production.engines.bayesian_estimator import (
    ICBayesianEstimator,
    ICPosterior,
    VolBayesianEstimator,
    VolPosterior,
    StrategyEstimatorRegistry,
    compute_confidence_weight,
)

from production.engines.risk_engine import (
    RiskEngine,
    PositionSizeRequest,
    PositionSizeResult,
    DrawdownTracker,
    PortfolioRiskTracker,
    RiskCheckResult,
    DrawdownLevel,
    compute_cvar,
    cornish_fisher_var,
)

from production.engines.portfolio_engine import (
    PortfolioEngine,
    PortfolioAllocation,
    StressTestResult,
    equal_risk_contribution_weights,
    ledoit_wolf_covariance,
    ewma_covariance,
    regime_conditioned_covariance,
    diversification_ratio,
    run_correlation_stress_tests,
)

__all__ = [
    # Direction
    "SignalDirection",

    # Engines
    "MeanReversionEngine",
    "TrendFollowingEngine",
    "VolatilityBreakoutEngine",
    "RegimeEngine",
    "StatisticalArbitrageEngine",

    # Signals
    "MeanReversionSignal",
    "TrendSignal",
    "BreakoutSignal",
    "StatArbSignal",

    # Regime
    "RegimeState",
    "StrategyScales",
    "VolatilityLevel",
    "TrendState",
    "get_regime_scales",
    "normalize_strategy_scales",
    "ALLOCATION_MATRIX",
    "cusum_structural_break",

    # Stat arb
    "CointegrationResult",
    "engle_granger_cointegration",
    "STAT_ARB_ALLOCATION_PCT",
    "APPROVED_PAIRS",

    # Indicators (for research use)
    "rolling_zscore",
    "rolling_autocorrelation",
    "compute_atr",
    "session_volatility_profile",
    "exponential_moving_average",
    "donchian_channel",
    "momentum_12m",
    "atr_volatility_ratio",
    "bollinger_band_width",
]
