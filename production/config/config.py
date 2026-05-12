"""
config.py — Configuration Manager
Loads, validates, and provides access to all system configuration.

Design principles:
- Risk limits are NEVER read from a database — only from committed YAML files
- Environment variables are resolved at load time (credentials never in YAML)
- Configuration is validated at startup; missing critical fields raise immediately
- A frozen config object is passed to all components — no runtime modification
"""

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DIRECTORY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent
LIVE_CONFIG_PATH = CONFIG_DIR / "live_config.yaml"
INSTRUMENTS_PATH = CONFIG_DIR / "instruments.yaml"
RISK_LIMITS_PATH = CONFIG_DIR / "risk_limits.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLE RESOLUTION
# Handles ${VAR_NAME} syntax in YAML values
# ─────────────────────────────────────────────────────────────────────────────
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(obj: Any) -> Any:
    """
    Recursively resolve ${ENV_VAR} references in config values.
    Raises EnvironmentError if a required variable is not set.
    """
    if isinstance(obj, str):
        def replace(match):
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                raise EnvironmentError(
                    f"Required environment variable '{var_name}' is not set. "
                    f"Set it on the VPS before starting the system."
                )
            return value
        return _ENV_VAR_PATTERN.sub(replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its content as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _require(cfg: dict, *keys: str, context: str = "") -> None:
    """Assert that all keys are present and non-empty in a dict."""
    for key in keys:
        if key not in cfg or cfg[key] in (None, "", [], {}):
            raise ValueError(f"Missing required config key '{key}' in {context}")


def _validate_live_config(cfg: dict) -> None:
    """Validate live_config.yaml structure and critical fields."""
    _require(cfg, "mt5", "active_instruments", "active_strategies",
             "timeframes", "regime", "portfolio", "monitoring", "deployment",
             context="live_config.yaml")

    # Deployment must be filled in
    deploy = cfg["deployment"]
    if deploy.get("environment") == "live":
        if not deploy.get("deployed_by"):
            raise ValueError(
                "live_config.yaml: 'deployed_by' must be set before live deployment. "
                "A named human reviewer is required (Section 10.3)."
            )
        if not deploy.get("deployed_at"):
            raise ValueError(
                "live_config.yaml: 'deployed_at' must be set before live deployment."
            )

    # Active instruments must be non-empty
    if not cfg["active_instruments"]:
        raise ValueError("live_config.yaml: 'active_instruments' list is empty.")


def _validate_risk_limits(cfg: dict) -> None:
    """Validate risk_limits.yaml — critical safety checks."""
    _require(cfg, "trade", "strategy", "portfolio", "kill_switch",
             "volatility_target", context="risk_limits.yaml")

    port = cfg["portfolio"]
    kill = cfg["kill_switch"]

    # Internal limits must be below hard limits
    if port["max_daily_loss_pct"] >= port["drawdown_hard_limit_pct"]:
        raise ValueError(
            "risk_limits: max_daily_loss_pct must be less than drawdown_hard_limit_pct"
        )

    if kill["daily_loss_trigger_pct"] > port["max_daily_loss_pct"] * 2:
        logger.warning(
            "Kill switch trigger is more than 2x the daily loss limit — "
            "consider tightening the kill switch threshold."
        )

    # Stat arb must be zero at launch
    if cfg["strategy"].get("stat_arb_allocation_pct", 0) > 0:
        raise ValueError(
            "risk_limits: stat_arb_allocation_pct must be 0.0 until leg execution "
            "limitations are resolved (Section 3, Engine 5)."
        )


def _validate_instruments(cfg: dict) -> None:
    """Validate instruments.yaml — check required fields per instrument."""
    required_fields = [
        "asset_class", "lot_size", "point_value", "contract_size",
        "max_spread_points", "max_bar_range", "atr_period",
        "mean_reversion_threshold", "mean_reversion_window",
    ]
    for symbol, spec in cfg.get("instruments", {}).items():
        for field_name in required_fields:
            if field_name not in spec:
                raise ValueError(
                    f"instruments.yaml: '{symbol}' missing required field '{field_name}'"
                )


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM CONFIG DATACLASS
# Provides typed access to all configuration sections
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SystemConfig:
    """
    Immutable system configuration object.
    Passed to all components at startup — never modified at runtime.
    """
    live: dict
    instruments: dict
    risk: dict

    @property
    def mt5(self) -> dict:
        return self.live["mt5"]

    @property
    def active_instruments(self) -> list[str]:
        return self.live["active_instruments"]

    @property
    def active_strategies(self) -> dict:
        return self.live["active_strategies"]

    @property
    def timeframes(self) -> dict:
        return self.live["timeframes"]

    @property
    def regime_params(self) -> dict:
        return self.live["regime"]

    @property
    def bayesian_params(self) -> dict:
        return self.live["bayesian"]

    @property
    def portfolio_params(self) -> dict:
        return self.live["portfolio"]

    @property
    def monitoring_params(self) -> dict:
        return self.live["monitoring"]

    @property
    def deployment(self) -> dict:
        return self.live["deployment"]

    @property
    def trade_limits(self) -> dict:
        return self.risk["trade"]

    @property
    def strategy_limits(self) -> dict:
        return self.risk["strategy"]

    @property
    def portfolio_limits(self) -> dict:
        return self.risk["portfolio"]

    @property
    def kill_switch_params(self) -> dict:
        return self.risk["kill_switch"]

    @property
    def vol_target_params(self) -> dict:
        return self.risk["volatility_target"]

    @property
    def kelly_params(self) -> dict:
        return self.risk["kelly"]

    @property
    def decay_params(self) -> dict:
        return self.risk["decay_monitor"]

    @property
    def heartbeat_params(self) -> dict:
        return self.risk["heartbeat"]

    @property
    def drawdown_schedule(self) -> list[dict]:
        return self.risk["drawdown_response"]

    def instrument(self, symbol: str) -> dict:
        """Return spec for a specific instrument. Raises if not found."""
        spec = self.instruments.get("instruments", {}).get(symbol)
        if spec is None:
            raise KeyError(f"Instrument '{symbol}' not found in instruments.yaml")
        return spec

    def is_live(self) -> bool:
        return self.deployment.get("environment") == "live"

    def is_paper(self) -> bool:
        return self.deployment.get("environment") == "paper"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_config(
    live_path: Path = LIVE_CONFIG_PATH,
    instruments_path: Path = INSTRUMENTS_PATH,
    risk_path: Path = RISK_LIMITS_PATH,
    resolve_env: bool = True,
) -> SystemConfig:
    """
    Load, validate, and return an immutable SystemConfig object.

    Parameters
    ----------
    live_path : Path to live_config.yaml
    instruments_path : Path to instruments.yaml
    risk_path : Path to risk_limits.yaml
    resolve_env : If True, resolve ${ENV_VAR} references (default True).
                  Set False in testing to avoid requiring env vars.

    Returns
    -------
    SystemConfig — frozen dataclass with typed access to all config sections.

    Raises
    ------
    FileNotFoundError — if any config file is missing
    ValueError — if required fields are missing or invalid
    EnvironmentError — if required environment variables are not set
    """
    logger.info("Loading system configuration...")

    live_cfg = _load_yaml(live_path)
    instruments_cfg = _load_yaml(instruments_path)
    risk_cfg = _load_yaml(risk_path)

    # Resolve environment variables
    if resolve_env:
        live_cfg = _resolve_env_vars(live_cfg)
        # instruments and risk don't contain credentials — no env resolution needed

    # Validate each config section
    _validate_live_config(live_cfg)
    _validate_instruments(instruments_cfg)
    _validate_risk_limits(risk_cfg)

    config = SystemConfig(
        live=live_cfg,
        instruments=instruments_cfg,
        risk=risk_cfg,
    )

    env = config.deployment.get("environment", "unknown")
    logger.info(
        f"Configuration loaded — environment={env}, "
        f"instruments={len(config.active_instruments)}, "
        f"framework_version={config.deployment.get('framework_version', '?')}"
    )
    return config


# ─────────────────────────────────────────────────────────────────────────────
# CLI VALIDATION — run directly to check config before deployment
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    try:
        # Don't resolve env vars in CLI check — just validate structure
        cfg = load_config(resolve_env=False)
        print("\n✓ Configuration valid")
        print(f"  Environment : {cfg.deployment.get('environment')}")
        print(f"  Instruments : {', '.join(cfg.active_instruments)}")
        print(f"  Framework   : v{cfg.deployment.get('framework_version')}")
        active = [k for k, v in cfg.active_strategies.items() if v]
        print(f"  Strategies  : {', '.join(active)}")
        print(f"  Daily limit : {cfg.portfolio_limits['max_daily_loss_pct']*100:.1f}%")
        print(f"  Kill switch : {cfg.kill_switch_params['daily_loss_trigger_pct']*100:.1f}% daily loss")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
