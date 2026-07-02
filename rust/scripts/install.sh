#!/usr/bin/env bash
set -euo pipefail

# TurboQuant Installation Script
# Supports: Debian 12/13, Ubuntu 22.04+, Fedora 40+, Arch Linux

PREFIX="/usr/local"
WITH_CUDA=""
WITH_DAEMON="yes"
WITH_SYSTEMD="yes"
USER_MODE=""
NON_INTERACTIVE=""
DRY_RUN=""
LOG_FILE="/var/log/turboquant-install.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --prefix) PREFIX="$2"; shift 2 ;;
            --no-cuda) WITH_CUDA="no" ; shift ;;
            --no-daemon) WITH_DAEMON="no" ; shift ;;
            --no-systemd) WITH_SYSTEMD="no" ; shift ;;
            --user) USER_MODE="yes" ; PREFIX="$HOME/.local" ; shift ;;
            --yes) NON_INTERACTIVE="yes" ; shift ;;
            --dry-run) DRY_RUN="yes" ; shift ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --prefix PATH     Install prefix (default: /usr/local)"
                echo "  --no-cuda         Skip CUDA support"
                echo "  --no-daemon       Skip daemon installation"
                echo "  --no-systemd      Skip systemd integration"
                echo "  --user            Install in ~/.local/bin"
                echo "  --yes             Non-interactive mode"
                echo "  --dry-run         Show what would be done"
                exit 0
                ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="${VERSION_ID:-}"
        log "Detected OS: $NAME $VERSION_ID"
    else
        log "ERROR: Cannot detect OS"
        exit 1
    fi
}

install_deps() {
    case "$OS_ID" in
        debian|ubuntu)
            if [ "$DRY_RUN" = "yes" ]; then
                log "[DRY RUN] apt-get update && apt-get install -y build-essential pkg-config libssl-dev curl git"
            else
                apt-get update -qq
                apt-get install -y -qq build-essential pkg-config libssl-dev curl git
            fi
            ;;
        fedora)
            if [ "$DRY_RUN" = "yes" ]; then
                log "[DRY RUN] dnf install -y gcc gcc-c++ make pkgconfig openssl-devel curl git"
            else
                dnf install -y gcc gcc-c++ make pkgconfig openssl-devel curl git
            fi
            ;;
        arch)
            if [ "$DRY_RUN" = "yes" ]; then
                log "[DRY RUN] pacman -S --noconfirm base-devel openssl curl git"
            else
                pacman -S --noconfirm base-devel openssl curl git
            fi
            ;;
        *)
            log "ERROR: Unsupported OS: $OS_ID"
            exit 1
            ;;
    esac
}

install_rust() {
    if command -v rustup &>/dev/null; then
        log "Rustup already installed"
    else
        if [ "$NON_INTERACTIVE" = "yes" ]; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
        else
            echo "Rust is not installed. Install via rustup? [Y/n]"
            read -r answer
            if [[ "$answer" =~ ^[Nn] ]]; then
                log "ERROR: Rust is required"
                exit 1
            fi
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
        fi
    fi
    source "$HOME/.cargo/env"
    rustup toolchain install stable
    rustup default stable
    log "Rust: $(rustc --version)"
}

detect_cuda() {
    if [ "$WITH_CUDA" = "no" ]; then
        return
    fi
    if command -v nvcc &>/dev/null; then
        log "CUDA detected: $(nvcc --version | grep release)"
        WITH_CUDA="yes"
    else
        log "CUDA not found — skipping GPU backend"
        WITH_CUDA="no"
    fi
}

build_turboquant() {
    # The CPU backend is a regular workspace crate — there is no `cpu`
    # cargo feature. The CUDA backend is not implemented yet, so nothing
    # extra is built even when CUDA is detected.
    if [ "$WITH_CUDA" = "yes" ]; then
        log "CUDA detected, but the CUDA backend is not implemented yet — building CPU-only"
    fi
    if [ "$DRY_RUN" = "yes" ]; then
        log "[DRY RUN] cargo build --release --workspace"
    else
        source "$HOME/.cargo/env"
        cargo build --release --workspace
    fi
}

install_binaries() {
    if [ "$DRY_RUN" = "yes" ]; then
        log "[DRY RUN] cp target/release/turboquant $PREFIX/bin/"
        log "[DRY RUN] cp target/release/turboquant-daemon $PREFIX/bin/"
        return
    fi
    mkdir -p "$PREFIX/bin"
    cp target/release/turboquant "$PREFIX/bin/"
    chmod 755 "$PREFIX/bin/turboquant"
    log "Installed turboquant CLI to $PREFIX/bin/turboquant"

    if [ "$WITH_DAEMON" = "yes" ]; then
        cp target/release/turboquant-daemon "$PREFIX/bin/"
        chmod 755 "$PREFIX/bin/turboquant-daemon"
        log "Installed turboquant-daemon to $PREFIX/bin/turboquant-daemon"
    fi
}

install_systemd() {
    if [ "$WITH_SYSTEMD" != "yes" ] || [ "$USER_MODE" = "yes" ]; then
        return
    fi

    if ! command -v systemctl &>/dev/null; then
        log "systemd not found — skipping service installation"
        return
    fi

    if [ "$DRY_RUN" = "yes" ]; then
        log "[DRY RUN] cp scripts/systemd/turboquant.service /etc/systemd/system/"
        return
    fi

    # Create user
    if ! id turboquant &>/dev/null; then
        useradd -r -s /usr/sbin/nologin -d /var/lib/turboquant turboquant
        log "Created system user: turboquant"
    fi

    cp scripts/systemd/turboquant.service /etc/systemd/system/
    systemctl daemon-reload
    log "Installed systemd unit: turboquant.service"
    log "To enable and start: systemctl enable --now turboquant"
}

install_man() {
    if [ "$DRY_RUN" = "yes" ]; then
        log "[DRY RUN] Install man page"
        return
    fi
    mkdir -p "$PREFIX/share/man/man1"
    # Generate man page from CLI
    if "$PREFIX/bin/turboquant" --help 2>/dev/null; then
        log "Man page will be generated by clap_mangen"
    fi
}

install_completions() {
    if [ "$DRY_RUN" = "yes" ]; then
        log "[DRY RUN] Install shell completions"
        return
    fi

    # Bash
    if [ -d /usr/share/bash-completion/completions ]; then
        mkdir -p /usr/share/bash-completion/completions
        "$PREFIX/bin/turboquant" generate-completions bash > /usr/share/bash-completion/completions/turboquant 2>/dev/null || true
    fi
    # Zsh
    if [ -d /usr/share/zsh/site-functions ]; then
        mkdir -p /usr/share/zsh/site-functions
        "$PREFIX/bin/turboquant" generate-completions zsh > /usr/share/zsh/site-functions/_turboquant 2>/dev/null || true
    fi
    # Fish
    if [ -d /usr/share/fish/completions ]; then
        mkdir -p /usr/share/fish/completions
        "$PREFIX/bin/turboquant" generate-completions fish > /usr/share/fish/completions/turboquant.fish 2>/dev/null || true
    fi
    log "Installed shell completions"
}

main() {
    parse_args "$@"

    if [ "$EUID" -ne 0 ] && [ "$USER_MODE" != "yes" ]; then
        log "ERROR: Run as root or use --user flag"
        exit 1
    fi

    mkdir -p "$(dirname "$LOG_FILE")"
    log "=== TurboQuant Installation ==="
    log "Prefix: $PREFIX"

    detect_os
    install_deps
    install_rust
    detect_cuda
    build_turboquant
    install_binaries
    install_systemd
    install_man
    install_completions

    log "=== Installation Complete ==="
    echo ""
    echo "TurboQuant installed successfully!"
    echo "  CLI: $PREFIX/bin/turboquant"
    echo "  Try: turboquant info"
    if [ "$WITH_SYSTEMD" = "yes" ] && [ "$USER_MODE" != "yes" ]; then
        echo "  Daemon: systemctl enable --now turboquant"
    fi
}

main "$@"
