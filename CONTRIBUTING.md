# Contributing to TurboQuant

Thank you for your interest in contributing!

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Hadamard rotation support
fix: correct 3-bit packer overflow
docs: update algorithm documentation
test: add roundtrip property tests
```

## Pull Request Checklist

- [ ] Code compiles: `cargo check --workspace --all-features`
- [ ] Formatting: `cargo fmt --all -- --check`
- [ ] Linting: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] Tests pass: `cargo test --workspace --all-features`
- [ ] Doc compiles: `cargo doc --workspace --no-deps` (zero warnings)
- [ ] No `unwrap`/`expect` in production code
- [ ] Public API is documented with runnable examples

## Development Setup

```bash
cd rust/
cargo build --workspace
cargo test --workspace
```

## Project Structure

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.
