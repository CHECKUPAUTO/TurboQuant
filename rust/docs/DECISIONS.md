# Architecture Decision Records

## ADR-001: Use ndarray over nalgebra

**Date**: 2026-05-06
**Status**: Accepted

**Context**: Need linear algebra for rotation matrices and tensor operations.
**Options**: nalgebra (strong typing, SIMD), ndarray (flexible, NumPy-like).
**Decision**: ndarray — matches Python mental model, easier migration path.
**Consequences**: No compile-time dimension checking; runtime shape validation needed.

## ADR-002: rayon over custom thread pool

**Date**: 2026-05-06
**Status**: Accepted

**Context**: CPU parallelism for multi-head compression.
**Options**: rayon (work-stealing), custom thread pool, tokio blocking.
**Decision**: rayon — standard in Rust ecosystem, zero-config work stealing.
**Consequences**: May oversubscribe with nested parallelism; mitigated by `num_threads` config.

## ADR-003: cbindgen over manual FFI

**Date**: 2026-05-06
**Status**: Accepted

**Context**: Need C ABI for integration with llama.cpp and other C consumers.
**Options**: cbindgen (auto-generate from Rust), manual C headers.
**Decision**: cbindgen — avoids drift between implementation and header.
**Consequences**: Build dependency on cbindgen; header checked into repo for consumers without Rust.

## ADR-004: MSRV 1.83

**Date**: 2026-05-06
**Status**: Accepted

**Context**: Minimum supported Rust version.
**Decision**: 1.83 (November 2024) — stable in Debian 13 backports.
**Consequences**: Cannot use edition 2024 crates; need to pin older versions of some dependencies.
