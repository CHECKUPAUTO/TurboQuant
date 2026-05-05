#!/bin/bash
# TurboQuant Deployment Script
# Usage: sudo bash deploy/install.sh
set -euo pipefail

echo "=== TurboQuant Deployment ==="

# 1. Install systemd units
echo "[1/4] Installing systemd units..."
for unit in deploy/*.service; do
    cp "$unit" /etc/systemd/system/
    echo "  → $(basename $unit)"
done

# 2. Reload systemd
echo "[2/4] Reloading systemd..."
systemctl daemon-reload

# 3. Enable and restart services
echo "[3/4] Enabling services..."
for unit in turboquant-proxy.service turboquant-watch.service turboquant-agent.service; do
    systemctl enable "$unit" 2>/dev/null || true
    systemctl restart "$unit" 2>/dev/null || echo "  (skipping $unit — may need manual start)"
done

# 4. Verify
echo "[4/4] Checking status..."
for unit in turboquant-proxy.service turboquant-watch.service turboquant-agent.service; do
    echo -n "  $unit: "
    systemctl is-active "$unit" 2>/dev/null || echo "inactive"
done

echo "=== Done ==="
echo "Set OLLAMA_HOST=http://127.0.0.1:11435 to use TurboQuant proxy"
