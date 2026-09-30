# LION federation routing — glitchlab

This repository is a federated LION component: **Evolution Compiler / delta and invariant analysis**.

## Bounded architecture discovery

1. Read `cyber-lion.repository.json` for the local machine-readable role, capabilities, architecture exports/imports and dependency route.
2. Read only the local architecture artifacts listed in `architecture_knowledge.architecture_artifacts` that are relevant to the task.
3. For federation-wide architecture, semantic ownership, formalization, currentness or roadmap decisions, route to `DonkeyJJLove/ai_platform` → `LION/architecture/v1_5/README.md`.

## Invariants

- Local documentation owns local repository semantics; it does not become a parallel owner of global LION architecture.
- `DOCUMENTATION != AUTHORITY`, `RAG != LIVE_TRUTH`, `HISTORY != CURRENT_STATE`, `TARGET != IMPLEMENTATION`.
- Reacquire exact default-branch HEAD/TREE before claiming currentness.
- Changes to semantic exports/imports, architecture artifacts, repository role or dependencies require a matching update to `cyber-lion.repository.json`.
- A local repository change may invalidate global architecture projections; do not silently assume `ai_platform` documentation remains current.
- Do not copy the full federation architecture into this repository.
