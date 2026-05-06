#!/usr/bin/env bash
set -euo pipefail

PURGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --purge) PURGE="yes"; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "This will remove TurboQuant from your system."
if [ "$PURGE" = "yes" ]; then
    echo "  --purge: will also remove /var/lib/turboquant and /etc/turboquant"
fi
echo -n "Continue? [y/N] "
read -r answer
if [[ ! "$answer" =~ ^[Yy] ]]; then
    echo "Aborted."
    exit 0
fi

# Stop service
if systemctl is-active --quiet turboquant 2>/dev/null; then
    systemctl stop turboquant
    systemctl disable turboquant
fi

# Remove unit
rm -f /etc/systemd/system/turboquant.service
systemctl daemon-reload 2>/dev/null || true

# Remove binaries
rm -f /usr/local/bin/turboquant
rm -f /usr/local/bin/turboquant-daemon
rm -f ~/.local/bin/turboquant
rm -f ~/.local/bin/turboquant-daemon

# Remove completions
rm -f /usr/share/bash-completion/completions/turboquant
rm -f /usr/share/zsh/site-functions/_turboquant
rm -f /usr/share/fish/completions/turboquant.fish

# Remove man
rm -f /usr/local/share/man/man1/turboquant.1.gz

# Remove user
if id turboquant &>/dev/null; then
    userdel turboquant
fi

if [ "$PURGE" = "yes" ]; then
    rm -rf /var/lib/turboquant
    rm -rf /etc/turboquant
fi

echo "TurboQuant uninstalled."
