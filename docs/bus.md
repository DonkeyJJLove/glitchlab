# 02 — GLX BUS (tiles, guards, HUD)

## git-analytics tile
**Rola:** Parsuje `.glx/commit_analysis.json`; emituje `git.delta.ready`  
**Kontrakt zdarzenia (szablon):**
```json
{
  "type": "git.delta.ready",
  "commit": "Δ",
  "delta": { "files": "Δ" },
  "summary": { "add": "Δ", "del": "Δ" }
}
```

## code-ast service
**Rola:** Buduje AST (wg Δ/całość); emituje `code.ast.built`  
```json
{
  "type": "code.ast.built",
  "commit": "Δ",
  "scope": ["Δ"],
  "metrics": { "nodes": "Δ", "functions": "Δ", "classes": "Δ" }
}
```

## refactor-planner
**Rola:** Plan refaktorów; emituje `refactor.plan.ready`  
```json
{
  "type": "refactor.plan.ready",
  "commit": "Δ",
  "items": [{"path":"Δ","action":"Δ","rationale":"Δ"}]
}
```

## validators I1–I4
**Rola:** Bramki jakości (fail-closed)  
```json
{
  "type": "fail-closed",
  "commit": "Δ",
  "stage": "I1|I2|I3|I4",
  "errors": [{"code":"Δ","msg":"Δ","path":"Δ"}]
}
```

## HUD/Reports
**Rola:** Konsoliduje i publikuje do EGDB (`publish`)
