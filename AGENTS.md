# TurboQuant Agent Bootstrap Contract

Before autonomous coding, codec/format changes, storage or quality claims, model conversion, cross-repository integration, PR creation, or merge decisions, read:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/TURBOQUANT_ECOSYSTEM_ROADMAP.yaml
```

If the roadmap cannot be fetched or read, fail closed for major format, codec-semantic, model-conversion, quality-promotion, cross-repository integration, or merge decisions. Read-only diagnosis is allowed.

## Repository role

TurboQuant owns the specialized codec semantics, canonical low-bit format, storage accounting, CPU/CUDA codec backends, and TurboQuant-specific model conversion. It does not own SLHAv2 cache policy, ElasticXxx representation-selection runtime, SciRust general tensor semantics, or NNIS model runtime policy.

The maintained implementation is the Rust tree. The legacy Python prototype is historical/reference material, not a second independently evolving semantic source.

Synthetic round-trip SNR is codec evidence, not real-LLM quality evidence. Storage claims must include all required scale/correction/layout overhead. Real-model claims require matched baseline/model/tokenizer/workload measurements.

SLHAv2 already contains a TQ3 port: cross-repository work must prevent the two definitions from silently diverging and should use versioned shared fixtures/contracts.

Required CI must be green on the exact PR head before merge.

Reread the roadmap at every session start, before format/math/conversion changes, before SLHAv2/ElasticXxx/SciRust/NNIS integration, after evidence changes, and before relevant PR/merge decisions.

Do not merge the roadmap itself into `main` unless the user explicitly requests it.
