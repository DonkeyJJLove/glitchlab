# GlitchLab

GlitchLab is the LION federation component responsible for evolution compilation, delta normalization, invariant evaluation and structural-change analysis.

This root entrypoint is intentionally concise. Global LION architecture is owned by DonkeyJJLove/ai_platform; local architecture and operating rules remain in this repository.

## Read order

1. AGENTS.md — bounded federation routing.
2. cyber-lion.repository.json — machine-readable role, capabilities and semantic imports/exports.
3. docs/10_architecture.md — local architecture.
4. src/README.md — experimental application/platform description.
5. mosaic/README.md — AST/Mosaic research subsystem.
6. bench/README.md — benchmark material.

## Packaging note

pyproject.toml uses this README.md as the package long description. Package/layout consolidation requires a separate import/entrypoint migration with tests.

## Current cleanup boundary

The tracked temp/ hook family and parallel src/, analysis/, core/, glx/, mosaic/ and delta/ trees remain under reconciliation. They are not deleted by naming standardization alone.

DOCUMENTATION != AUTHORITY. HISTORY != CURRENT_STATE. UNKNOWN > DELETE.
