# GlitchLab Process Guard

This repository uses a delta-first maintenance contract aligned with the wider DonkeyJJLove research process.

## Invariants

```text
SOURCE ≠ GENERATED STATE
MODEL PROPOSAL ≠ AUTHORITY
CRITICAL CHANGE ⇒ RECONSTRUCTABLE PROVENANCE
LOCAL PASS ≠ PATH SAFE
```

## Review loop

```text
baseline
→ Δ inventory
→ AST / dependency / security probe
→ invariant check
→ patch
→ tests
→ SAST
→ delta report
→ merge / reject
```

Operational lenses:

- `_neuro` / EEG-like state model: baseline, delta, burst, coupling, drift, recovery;
- textual lithography: source → semantics → mandate → execution → observation → revision;
- `Ostrze–Cierpliwość`: generated fixes remain bounded by Guard and tests;
- `Próg–Przejście`: α/β/Z decisions must remain explicit and observable;
- `Rdzeń–Peryferia`: local environments, caches and build artifacts are not source code.

## Repository-specific checks

1. No `.env.local`, virtualenv, cache or `*.egg-info` state in versioned source.
2. `pyproject.toml`, actual package layout and README entrypoints must agree.
3. A generated repair is never accepted without an independent test/invariant gate.
4. SAST findings are converted into generalized regression conditions when possible.
5. Any change to AST↔Mosaic invariants requires a before/after delta artifact.

For the full cross-repository protocol see `writeups/PROCESS_GUARD.md`.
