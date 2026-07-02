#!/usr/bin/env bash
set -euo pipefail

CHANNEL="stable"
ROLLBACK=""
BACKUP_DIR="/var/lib/turboquant/backups"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --channel) CHANNEL="$2"; shift 2 ;;
        --rollback) ROLLBACK="yes"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ "$ROLLBACK" = "yes" ]; then
    log "Rolling back..."
    if [ -f "$BACKUP_DIR/turboquant.prev" ]; then
        cp "$BACKUP_DIR/turboquant.prev" /usr/local/bin/turboquant
        log "Rolled back to previous version"
    else
        log "No backup found"
        exit 1
    fi
    exit 0
fi

# Backup current version
mkdir -p "$BACKUP_DIR"
if [ -f /usr/local/bin/turboquant ]; then
    cp /usr/local/bin/turboquant "$BACKUP_DIR/turboquant.prev"
fi

# Fetch latest
if git rev-parse --git-dir > /dev/null 2>&1; then
    git fetch --tags
    LATEST=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    if [ -n "$LATEST" ]; then
        log "Latest tag: $LATEST"
        git checkout "$LATEST"
    else
        git pull origin main
    fi
else
    log "Not in git repo — please pull manually"
    exit 1
fi

# Rebuild
source "$HOME/.cargo/env"
cargo build --release --workspace

# Atomically replace
cp target/release/turboquant /usr/local/bin/turboquant.new
mv /usr/local/bin/turboquant.new /usr/local/bin/turboquant
cp target/release/turboquant-daemon /usr/local/bin/turboquant-daemon.new
mv /usr/local/bin/turboquant-daemon.new /usr/local/bin/turboquant-daemon

# Restart if running
if systemctl is-active --quiet turboquant 2>/dev/null; then
    systemctl restart turboquant
    log "Service restarted"
fi

log "Update complete: $(turboquant --version)"
