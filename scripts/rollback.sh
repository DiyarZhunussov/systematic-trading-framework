#!/bin/bash
# scripts/rollback.sh — Emergency rollback to last stable configuration
# Implements Section 9.6 of the framework
#
# Usage:
#   ./scripts/rollback.sh                    # Full rollback (close positions + restore config)
#   ./scripts/rollback.sh --config-only      # Restore config without closing positions
#   ./scripts/rollback.sh --positions-only   # Close positions without restoring config
#
# Trigger conditions (Section 9.6):
#   - Any live drawdown > 3% in first 5 days of a new deployment
#   - Manual invocation by human operator after investigation
#   - Called by automated monitoring if deployment_health_check fails
#
# Requirements:
#   - TRADING_SYSTEM_ROOT env var set, OR script run from project root
#   - Python environment with MetaTrader5 installed
#   - MT5_LOGIN, MT5_PASSWORD, MT5_SERVER env vars set

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TRADING_SYSTEM_ROOT:-$(dirname "$SCRIPT_DIR")}"
PRODUCTION_DIR="$ROOT/production"
CONFIG_DIR="$PRODUCTION_DIR/config"
LOG_DIR="$PRODUCTION_DIR/logs/system"
BACKUP_DIR="$CONFIG_DIR/backup"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$LOG_DIR/rollback_$(date -u +%Y%m%d_%H%M%S).log"

# Flags
CLOSE_POSITIONS=true
RESTORE_CONFIG=true

# Parse arguments
for arg in "$@"; do
  case $arg in
    --config-only)
      CLOSE_POSITIONS=false
      ;;
    --positions-only)
      RESTORE_CONFIG=false
      ;;
    --help)
      echo "Usage: $0 [--config-only | --positions-only]"
      exit 0
      ;;
  esac
done

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$TIMESTAMP] $*"; }
log_error() { echo "[$TIMESTAMP] ERROR: $*" >&2; }

# ── Start ─────────────────────────────────────────────────────────────────────
log "============================================================"
log "ROLLBACK INITIATED"
log "Root: $ROOT"
log "Close positions: $CLOSE_POSITIONS"
log "Restore config: $RESTORE_CONFIG"
log "============================================================"

# ── Step 1: Close all positions ───────────────────────────────────────────────
if [ "$CLOSE_POSITIONS" = true ]; then
  log "Step 1: Closing all open positions..."
  python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get('TRADING_SYSTEM_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from production.config.config import load_config
    from production.execution.mt5_bridge import MT5Bridge
    config = load_config()
    bridge = MT5Bridge(config)
    if bridge.connect():
        positions = bridge.get_open_positions()
        print(f"  Found {len(positions)} open positions")
        for pos in sorted(positions, key=lambda p: p.get('profit', 0)):
            success = bridge.close_position(pos['ticket'], reason='rollback')
            status = "closed" if success else "FAILED"
            print(f"  ticket={pos['ticket']} {pos['symbol']} -> {status}")
        bridge.disconnect()
        print("  Position closure complete")
    else:
        print("  WARNING: MT5 connection failed — manual position closure required")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)
PYEOF
  if [ $? -ne 0 ]; then
    log_error "Position closure failed or MT5 unavailable"
    log_error "MANUAL ACTION REQUIRED: Close all positions in MT5 terminal"
    log_error "Continuing with config restore..."
  else
    log "Step 1: Positions closed successfully"
  fi
else
  log "Step 1: Skipped (--config-only)"
fi

# ── Step 2: Stop trading services ─────────────────────────────────────────────
log "Step 2: Stopping trading services..."
if command -v systemctl &>/dev/null; then
  if systemctl is-active --quiet trading-main 2>/dev/null; then
    systemctl stop trading-main
    log "  trading-main service stopped"
  else
    log "  trading-main service not running (or not a systemd service)"
  fi
else
  # Try to kill by process name
  pkill -f "production/main.py" 2>/dev/null && log "  Killed main.py process" || log "  No main.py process found"
fi

# ── Step 3: Restore previous configuration ────────────────────────────────────
if [ "$RESTORE_CONFIG" = true ]; then
  log "Step 3: Restoring previous configuration..."

  # Method A: git stash (preferred — preserves history)
  if [ -d "$ROOT/.git" ]; then
    cd "$ROOT"
    if git stash list | grep -q "stash"; then
      git stash pop
      log "  Restored config via git stash pop"
    else
      # Try to restore from last known good tag
      LAST_GOOD=$(git tag -l "deploy-*" | sort -V | tail -2 | head -1)
      if [ -n "$LAST_GOOD" ]; then
        git checkout "$LAST_GOOD" -- production/config/live_config.yaml
        log "  Restored config from tag: $LAST_GOOD"
      else
        log_error "  No git stash or deploy tags found — trying backup files"
      fi
    fi
  fi

  # Method B: backup file fallback
  BACKUP_FILE="$BACKUP_DIR/live_config.yaml.bak"
  if [ -f "$BACKUP_FILE" ]; then
    cp "$BACKUP_FILE" "$CONFIG_DIR/live_config.yaml"
    log "  Restored config from backup: $BACKUP_FILE"
  else
    log_error "  No backup config found at $BACKUP_FILE"
    log_error "  Manual config restore required"
  fi
else
  log "Step 3: Skipped (--positions-only)"
fi

# ── Step 4: Verify restored config ────────────────────────────────────────────
log "Step 4: Verifying restored configuration..."
python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get('TRADING_SYSTEM_ROOT', '.'))
try:
    from production.config.config import load_config
    cfg = load_config(resolve_env=False)
    print(f"  Config valid: environment={cfg.deployment.get('environment')}")
    print(f"  Framework version: {cfg.deployment.get('framework_version')}")
    print(f"  Instruments: {cfg.active_instruments}")
    sys.exit(0)
except Exception as e:
    print(f"  Config validation FAILED: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
  log_error "Config validation failed — system cannot restart safely"
  log_error "Manual investigation required before restart"
  exit 1
fi
log "Step 4: Config validation passed"

# ── Step 5: Restart services ──────────────────────────────────────────────────
log "Step 5: Services NOT auto-restarted after rollback"
log "  Reason: Manual human review required before resuming trading"
log "  To restart: systemctl start trading-main (or python production/main.py)"
log ""
log "  Required before restart:"
log "    1. Investigate the cause of rollback"
log "    2. Confirm positions are flat"
log "    3. Confirm config is correct"
log "    4. Human sign-off (update deployed_by and deployed_at in live_config.yaml)"

# ── Completion ─────────────────────────────────────────────────────────────────
log "============================================================"
log "ROLLBACK COMPLETE: $TIMESTAMP"
log "Log file: $LOG_FILE"
log "============================================================"
