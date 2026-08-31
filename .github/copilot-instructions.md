# TurboQuant repository agent instructions

Before repository changes, fetch and read the persistent off-main roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/TURBOQUANT_ECOSYSTEM_ROADMAP.yaml
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If the roadmap is unavailable, fail closed for major format, codec-semantic, model-conversion, quality-promotion, cross-repository integration, or merge decisions.

Preserve one canonical Rust codec semantics, exact storage accounting, explicit format versions, real-model evidence boundaries, and shared-version convergence with SLHAv2 rather than silent TQ3 divergence.
