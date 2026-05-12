# Institutional Quantitative Trading Framework
**Multi-Strategy Systematic Portfolio for Forex & Index CFDs — MetaTrader 5**
Framework Version: 2.0 | Implementation Version: 1.0.0

---

## ⚠️ Critical Prerequisites Before Any Live Trading

1. **Complete paper trading** — minimum 60 trading days per strategy (Section 10.1 Stage 4)
2. **Human adversarial reviewer** sign-off — named human must attempt to falsify each strategy (Section 10.3)
3. **DSR > 0.85** for all engines (> 0.95 for volatility breakout)
4. **PBO < 0.10** for all engines
5. **`environment: live`** in `live_config.yaml` requires `deployed_by` and `deployed_at` fields

---

## Project Structure

```
trading-system/
├── production/
│   ├── main.py                    ← Entry point — run this
│   ├── config/
│   │   ├── config.py              ← Config loader (validates all YAML)
│   │   ├── live_config.yaml       ← Runtime config (edit this)
│   │   ├── instruments.yaml       ← Per-instrument specs
│   │   └── risk_limits.yaml       ← Hard risk limits (never modify at runtime)
│   ├── data/
│   │   ├── feed_manager.py        ← MT5 data ingestion + in-memory cache
│   │   └── data_validator.py      ← OHLCV validation (bar + series)
│   ├── engines/
│   │   ├── regime_engine.py       ← ADX regime detection + crisis layer
│   │   ├── signal_engine_mean_reversion.py
│   │   ├── signal_engine_trend_following.py
│   │   ├── signal_engine_volatility_breakout.py
│   │   ├── signal_engine_stat_arb.py  ← ZERO ALLOCATION — research only
│   │   ├── bayesian_estimator.py  ← IC + vol Bayesian posteriors
│   │   ├── risk_engine.py         ← Position sizing + drawdown control
│   │   └── portfolio_engine.py    ← ERC + covariance + rebalancing
│   ├── execution/
│   │   ├── mt5_bridge.py          ← MT5 orders + fill tracking
│   │   └── order_manager.py       ← Trade lifecycle management
│   └── monitoring/
│       ├── heartbeat.py           ← Supervisor thread + kill switch
│       ├── decay_monitor.py       ← Alpha decay (CUSUM + 5 conditions)
│       ├── performance_monitor.py ← Sharpe, Sortino, Calmar, win rate
│       └── structured_logger.py   ← JSONL logs (trades/signals/risk/system)
├── research/                      ← Separate from production (never share code at runtime)
├── staging/                       ← Paper trading and shadow mode
├── tests/
│   ├── unit/
│   ├── integration/
│   └── chaos/                     ← Fault injection tests
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables (never hardcode credentials)
```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD=your_password
export MT5_SERVER=ICMarkets-Demo
export TELEGRAM_BOT_TOKEN=your_bot_token   # optional
export TELEGRAM_CHAT_ID=your_chat_id       # optional
```

### 3. Validate configuration
```bash
python production/config/config.py
```

### 4. Start in paper trading mode
```bash
# Ensure live_config.yaml has environment: paper
python production/main.py
```

### 5. Switch to live (after all prerequisites met)
```yaml
# In live_config.yaml:
deployment:
  environment: "live"
  deployed_by: "Your Name"
  deployed_at: "2025-01-01T00:00:00Z"
```

---

## VPS Requirements

- **Latency target**: sub-20ms round-trip to broker's matching engine
- **IC Markets / Pepperstone US** → Equinix NY4 (Secaucus, NJ)
- **IC Markets AU / Pepperstone AU** → Equinix LD4 (Slough, UK)
- **Asian brokers** → Equinix TY3 (Tokyo)
- Sub-1ms co-location is unnecessary for this strategy class

---

## Risk Limits Summary

| Level | Limit | Trigger |
|-------|-------|---------|
| Per trade | 0.5% account | Hard stop always set |
| Strategy daily | 1.5% account | No new entries |
| Portfolio daily | 2.0% account | All new entries blocked |
| Kill switch | 3.0% daily OR 8% drawdown | Close all positions |
| Prop firm buffer | 2× internal vs prop limit | Buffer for measurement lag |

---

## Kill Switch

The kill switch fires automatically on:
- Component heartbeat silent > 120 seconds
- MT5 terminal not responding
- Margin level < 200%
- Daily loss > 3% of account
- Drawdown > 8% of account

**Manual reset required** — no automatic restart after kill switch:
```python
supervisor.reset(authorised_by="Your Name")
risk_engine.reset_kill_switch(authorised_by="Your Name")
```

---

## Governance Rules (Section 10)

1. **No LLM review** for adversarial hypothesis testing or deployment decisions
2. **Named human adversarial reviewer** required for Stage 4 → Stage 6 promotion
3. **Statistical arbitrage**: zero allocation until leg execution prerequisites met
4. **Volatility breakout**: DSR > 0.95 required (vs 0.85 for other engines)
5. **Being close to a limit is NEVER justification for increasing risk**

---

## Log Files

Logs are written to `production/logs/` in JSONL format:
- `trades/YYYY-MM-DD.jsonl` — every trade open/close
- `signals/YYYY-MM-DD.jsonl` — every signal generated
- `risk/YYYY-MM-DD.jsonl` — drawdown, kill switch, decay alerts
- `system/YYYY-MM-DD.jsonl` — heartbeat, regime changes, performance

---

## Framework Reference

Based on: *Institutional Quantitative Trading Framework v2.0*
Adversarial review corrections applied:
- Bayesian IC prior revised: Beta(3,30) → Beta(2,30)
- TSMOM alpha decomposed via Kim, Tse & Wald (2016)
- Hurst exponent retired as primary regime input → ADX
- Gross leverage normalisation function added
- False-precision probability-of-success estimate removed
- LLM review prohibition and human adversarial reviewer mandate added
