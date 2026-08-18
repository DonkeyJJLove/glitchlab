# GlitchLab — AI-Native Enterprise Roadmap

Enterprise role: **Evolution Compiler** for software and organizational deltas.

GlitchLab should remain the system that answers:

> What actually changed, which contracts/invariants were affected, what risk was introduced, and may this delta proceed?

Its strongest current primitives are already aligned with that role:

```text
Δ-first analysis
AST ↔ Mosaic (Φ/Ψ)
I1–I4 invariant gates
α/β/ζ living thresholds
SAST-Bridge: NF → PQ → FixCandidate
BUS / EGDB / HUD observability
one loop → one artifact
```

## Target

Extend the compiler model beyond source code:

```text
SourceCodeDelta
AgentSpecDelta
SwarmSpecDelta
PolicyDelta
SchemaDelta
MemoryContractDelta
RepositoryManifestDelta
        ↓
normalization / fingerprint
        ↓
structural projection
        ↓
contract + security + observability checks
        ↓
EnterpriseDeltaReport
        ↓
ACCEPT / REVIEW / BLOCK
```

The result is an input to Cyber-Lion policy. GlitchLab does **not** grant runtime authority itself.

## Phase 1 — stabilize the existing compiler

- finish source/generated-state hygiene and package cleanup;
- prove all documented entrypoints in CI;
- reduce historical lint debt through a ratchet instead of hiding it;
- make generated-code execution fail-closed unless a real isolated execution provider is supplied;
- align current `docs/10_architecture.md`, invariant docs and actual package paths;
- keep `/spec` as local SSOT for GlitchLab-specific semantics.

## Phase 2 — general Enterprise Delta contract

Add a versioned neutral representation:

```text
EnterpriseDeltaToken {
  kind,
  entity_id,
  artifact,
  before,
  after,
  evidence_refs,
  risk_class,
  authority_delta,
  observability_delta
}
```

Initial token classes:

```text
ADD_AGENT
REMOVE_AGENT
MODIFY_AGENT_AUTHORITY
ADD_CAPABILITY
MODIFY_CAPABILITY_CONTRACT
ADD_SWARM_EDGE
REMOVE_SWARM_EDGE
MODIFY_POLICY
MODIFY_SCHEMA
MODIFY_MEMORY_RULE
```

## Phase 3 — adapters

Implement independent adapters for:

- Cyber-Lion `AgentSpec`,
- `MissionSpec` / `SwarmSpec` / `MosaicDelta`,
- capability descriptors,
- policies/gates,
- JSON Schema,
- repository manifests,
- HA2D-derived memory contracts.

Do not overload the existing local filter registry into a global capability registry.

## Phase 4 — enterprise invariants

Keep code-level I1–I4 and add a separate enterprise family:

```text
E1 identity continuity
E2 contract compatibility
E3 authority non-escalation
E4 provenance completeness
E5 observability preservation
E6 replayability
E7 bounded blast radius
E8 epistemic-status integrity
E9 context/memory separation
E10 swarm structural integrity
```

## Phase 5 — SAST-Bridge becomes a general security-finding bridge

Preserve the strong current pipeline:

```text
scanner/raw evidence
→ Normalized Finding
→ dedup
→ structural/Δ binding
→ prioritized queue
→ FixCandidate
→ tests/invariants
→ verification
```

Extend finding bindings to:

```text
source symbol
AgentSpec
capability
swarm edge
policy
execution domain
```

## Phase 6 — enterprise observability

EGDB should be able to ingest Cyber-Lion correlation/provenance identifiers and answer causal queries such as:

```text
mission
→ agent proposal
→ change delta
→ invariant result
→ gate
→ build/execution receipt
→ outcome
```

HUD should show not only "what failed" but **why this transition was allowed or blocked**.

## Phase 7 — self-healing under bounded authority

`FixCandidate` remains a proposal.

```text
finding
→ generalized missing invariant
→ candidate fix
→ negative tests
→ GlitchLab validation
→ Cyber-Lion authority gate
→ bounded execution
→ outcome
```

No consequential automatic patch should bypass the external authority plane.

## Integration contract

GlitchLab should eventually expose capabilities such as:

```text
structure.extract
change.delta.normalize
change.invariants.evaluate
security.findings.normalize
security.fix.propose
change.report.generate
```

Each capability declares input/output schema, side effects, required authority and observability events.

## Do not do

```text
GlitchLab score == production permission
visual similarity == security proof
model-generated patch == trusted patch
local registry == enterprise capability registry
adaptive threshold == unlimited normalization of drift
```

Security-critical threshold adaptation must use freeze-on-drift, bounded change and independent review.

## Enterprise references

Canonical enterprise architecture:

`https://github.com/DonkeyJJLove/ai_platform/tree/master/cyber_lion/enterprise`

GlitchLab remains independently deployable and testable. Federation is through typed contracts, not source-code copying.
