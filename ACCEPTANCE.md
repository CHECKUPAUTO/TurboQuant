# Acceptance Report — TurboQuant

Updated: 2026-07-02 (post-audit; supersedes the 2026-05-06 report, which
overstated several results — see `BUGS_FIXED.md`, bugs R1–R8).

## Build & Test

Verified on this branch, from `rust/`:

| Check | Status | Command |
|-------|--------|---------|
| cargo test (core + cpu) | ✅ verified | `cargo test -p turboquant-core -p turboquant-cpu` — 57 tests (turboquant-core: 30 unit + 15 integration + 8 doc; turboquant-cpu: 3 unit + 1 doc) |
| cargo test (workspace) | ✅ at time of writing | `cargo test --workspace` — 86 unit/integration + 9 doc tests passing (snapshot; several crates are under active development) |
| cargo clippy (core + cpu) | ✅ verified | `cargo clippy -p turboquant-core -p turboquant-cpu --all-targets -- -D warnings -W clippy::pedantic` |
| cargo fmt (core + cpu) | ✅ verified | `cargo fmt -p turboquant-core -p turboquant-cpu -- --check` |

Workspace-wide clippy/fmt/doc status for the other crates is tracked by
CI (`.github/workflows/ci.yml`); it is not re-asserted here.

## Quality Metrics (measured)

Measured by the test suite on standard-normal data, block size 64
(`rust/crates/turboquant-core/tests/`, `src/qjl.rs` tests):

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Round-trip SNR (3-bit + 1-bit correction, default) | >12 dB | ~19 dB | ✅ |
| Round-trip SNR (3-bit, no correction) | >12 dB | ~13 dB | ✅ |
| Compression ratio (no correction) | ≥5.0× | ~4.9× | ❌ slightly below target (scale overhead) |
| Compression ratio (default, correction persisted) | ≥5.0× | ~3.8× | ❌ correction costs 1 bit/value |
| Attention output vs float reference (SNR) | >12 dB | ~15 dB | ✅ |
| Attention output vs float reference (cosine) | ≥0.96 | ~0.985 | ✅ |

Note: the earlier report claimed 5.3× compression at SNR > 12 dB. That
combination is not achievable with this format: 5.33× ignores the
per-block scale overhead, and before the 2026-07 grid fix the measured
SNR was actually ~2.9 dB (see `BUGS_FIXED.md` R1).

## Component Status

| Component | Status |
|-----------|--------|
| turboquant-core | ✅ implemented and tested in this branch (grid, packing, KV storage, attention reference with measured stats) |
| turboquant-cpu | ✅ implemented (rayon parallelism; no explicit SIMD — the unused `wide` dependency was removed) |
| turboquant-cuda | 🚧 stub only — **no GPU code**; every entry point returns "not implemented" |
| turboquant-ffi | 🚧 work in progress — see crate docs |
| turboquant-py | 🚧 work in progress — see crate docs |
| turboquant-gguf | 🚧 work in progress (reader/writer) — see crate docs |
| turboquant-cli | 🚧 work in progress — see crate docs |
| turboquant-daemon | 🚧 work in progress — see crate docs |
| turboquant-bench | 🚧 criterion harness — see crate docs |

## Production Code Audit

`grep -rn "unwrap\|expect\|panic!\|todo!\|unimplemented!"` over
`turboquant-core`/`turboquant-cpu` production code: no matches outside
test modules. `KvBlock::store`/`store_correction` validate input lengths
and return `Result` instead of panicking on mismatched slices.

## Documentation

| Doc | Status |
|-----|--------|
| README.md | ✅ commands fixed (no fake `--features cpu`); gains table replaced with measured numbers |
| BUGS_FIXED.md | ✅ Python-port bugs 1–16 plus Rust-port bugs R1–R8; test mapping points at files that exist |
| CHANGELOG.md / CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md / LICENSE | ✅ present |
| docs/, mdBook | present; not re-verified in this audit |

## CI/CD

| Workflow | Status |
|----------|--------|
| ci.yml | ✅ fixed: removed nonexistent `--features cpu` matrix entry |
| release.yml | ✅ fixed: removed nonexistent `--features=cpu` flag |
| docs.yml | present; not re-verified in this audit |

## Verdict

Core algorithm crates (`turboquant-core`, `turboquant-cpu`) pass build,
tests, pedantic clippy, and fmt with **measured** quality numbers as
listed above. The remaining crates are in active development; CUDA
support does **not** exist yet. Not ready to claim v1.0.0 until the
in-progress crates land and the workspace-wide checks are re-run.
