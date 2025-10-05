# 10 — Załączniki i tagi automaty

- `<!-- @auto:... -->` — miejsce wstrzyknięcia danych z audytu/ZIP/analiz
- `<!-- @auto:diagram:XYZ -->` — renderuj dodatkowe grafy Mermaid
- `<!-- @auto:contract:EVENT -->` — wstaw JSON Schema zdarzenia

**Instrukcja integracji (pseudo):**
```
glx_doc_inject --file docs/GLX_ARCHITEKTURA_KATALOG.md   --source /mnt/data/AUDIT_YYYYMMDD.zip   --tag @auto:repo.glx.commit_analysis.preview   --value "$(jq '.delta | .files' .glx/commit_analysis.json)"
```
