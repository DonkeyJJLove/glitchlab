# 01 — Repo GLX

> Sekcja szablonowa — **bez analizy repo**. Zawiera znaczniki `@auto:` do uzupełnienia z Twoich narzędzi.

## Hooki

### `pre-commit.py`
- **Rola:** Inicjacja przygotowania Δ/HEAD → `pre-diff.py`
- **Wejścia:** staged changes
- **Wyjścia:** wywołanie `pre-diff.py`
- **Dane do wypełnienia (Δ):**
  - `<!-- @auto:repo.precommit.delta -->`
  - `<!-- @auto:repo.precommit.loc -->`

### `pre-diff.py`
- **Rola:** Buduje artefakt różnic i metadanych
- **Wyjścia:** → [`.glx/commit_analysis.json`](#glxcommitanalysisjson)

**Kontrakt (szablon):**
```json
{
  "commit": { "hash": "Δ", "parent": "Δ", "author": "Δ", "ts": "Δ" },
  "delta": { "files": "Δ", "summary": "Δ" },
  "head": { "branch": "Δ" }
}
```

### `.glx/commit_analysis.json`
- **Rola:** Artefakt repo z danymi Δ/HEAD
- **Konsumenci:** GA (git-analytics tile)

**Podgląd (opcjonalny):**
```
<!-- @auto:repo.glx.commit_analysis.preview -->
```

### `post-commit.py`
- **Rola:** Pakuje audyt → `AUDIT_*.zip`

### `AUDIT_*.zip`
- **Rola:** Artefakt archiwalny; raporty, logi, zrzuty
- **Lista zawartości (opcjonalna):**
```
<!-- @auto:repo.audit_zip.contents -->
```
