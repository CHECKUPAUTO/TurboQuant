# Installation

## One-liner

```bash
curl -fsSL https://raw.githubusercontent.com/CHECKUPAUTO/TurboQuant/main/rust/scripts/install.sh | bash
```

## From Source

```bash
git clone https://github.com/CHECKUPAUTO/TurboQuant
cd TurboQuant/rust
cargo build --release --workspace --features=cpu
sudo cp target/release/turboquant /usr/local/bin/
```

## System Requirements

- Rust 1.83+
- Linux x86_64 (Debian 12+, Ubuntu 22.04+, Fedora 40+, Arch)
- Optional: CUDA 11.4+ for GPU acceleration
