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

## Licensing of Contributions

TurboQuant is dual-licensed: free for noncommercial use under the
PolyForm Noncommercial License 1.0.0, with a separate commercial license
offered exclusively for use as a module of **CCOS** (TurboQuant and
[SLHAv2](https://github.com/CHECKUPAUTO/SLHAv2) are companion modules of
CCOS). See [LICENSING.md](LICENSING.md) for the full policy.

To preserve this dual-license model, external contributions are accepted
only under a Contributor License Agreement (CLA) that licenses the
contribution to the copyright holder for use under **both** the
noncommercial and the commercial license. By opening a pull request you
will be asked to accept the CLA before the contribution can be merged.
