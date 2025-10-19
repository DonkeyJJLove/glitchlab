### docs/90_playbooks.md
# Playbooks — SRE & DX operacyjne dla GlitchLab

Ten dokument zawiera **operacyjne playbooki** (SRE/DX) dla platformy GlitchLab. Każdy playbook jest zorganizowany w układzie: **kiedy stosować → wejścia → kroki → artefakty → rollback → uwagi**. Playbooki odwołują się do pojęć i mechanizmów opisanych w:

- `docs/11_spec_glossary.md` (S/H/Z, Δ, Φ/Ψ, I1–I4),
- `docs/12_invariants.md` (gates jakości),
- `docs/13_delta_algebra.md` (tokeny Δ, fingerprint),
- `docs/14_mosaic.md` (algorytm mozaiki i gałąź `mosaic/`),
- `docs/20_bus.md` i `docs/21_egdb.md` (BUS/EGDB),
- `docs/30_sast_bridge.md` (SAST Bridge),
- `docs/41_pipelines.md` (pipeline kroków i artefakty `.glx/*`),
- `docs/50_ci_ops.md` (hooki i workflow CI),
- `docs/60_security.md` (polityki bezpieczeństwa),
- `docs/70_observability.md` (telemetria, EGQL),
- `docs/80_release.md` (wydania, hotfix, artefakty).

> **Konwencje:**  
> - Wszystkie komendy uruchamiamy z katalogu głównego repo.  
> - Artefakty operacyjne lądują w `.glx/*` (audit, spec, delta, mozaiki).  
> - Zmienna `HEAD~k` oznacza zakres regresji; `BASE` to wspólny przodek.

---

## Spis playbooków

1. [Freeze/Unfreeze progów (α/β/Z) przy drifcie](#1-freezeunfreeze-progów-αβz-przy-drifcie)  
2. [Rollback patcha naruszającego I-gates](#2-rollback-patcha-naruszającego-i-gates)  
3. [Reindeksacja AST/Δ i rekalkulacja mozaiki](#3-reindeksacja-astΔ-i-rekalkulacja-mozaiki)  
4. [Rebuild EGDB, widoki i sanity check EGQL](#4-rebuild-egdb-widoki-i-sanity-check-egql)  
5. [Incident: Wyciek sekretu / znalezisko SAST CRITICAL](#5-incident-wyciek-sekretu--znalezisko-sast-critical)  
6. [Unstuck CI: pipeline czerwony, gating na Δ](#6-unstuck-ci-pipeline-czerwony-gating-na-Δ)  
7. [Perf regression: wzrost czasu kroków / spadek SSIM](#7-perf-regression-wzrost-czasu-kroków--spadek-ssim)  
8. [HUD triage: Delta Inspector & Spec Monitor](#8-hud-triage-delta-inspector--spec-monitor)  
9. [Release/hotfix procedura operacyjna](#9-releasehotfix-procedura-operacyjna)  
10. [Disaster Recovery: odtworzenie ze snapshotu `.glx` / `backup/AUDIT_*.zip`](#10-disaster-recovery-odtworzenie-ze-snapshotu-glx--backupaudit_zip)  
11. [Reindex presetów/filtrów i kwarantanna pluginów](#11-reindex-presetówfiltrów-i-kwarantanna-pluginów)  
12. [Playbook SRE: „Δ spike > Z” (nagły wzrost energii zmian)](#12-playbook-sre-δ-spike--z-nagły-wzrost-energii-zmian)

---

## 1) Freeze/Unfreeze progów (α/β/Z) przy drifcie

**Kiedy:** wykryto drift progów (Page-Hinkley/ADWIN), fluktuacje false-positives, seria PR-ów przekraczająca β bez merytorycznej przyczyny.

**Wejścia:** `.glx/spec_state.json`, `.glx/metrics.parquet`, logi z `Spec Monitor`.

**Kroki:**
```bash
# 1. Zapisz aktualny stan progów (snapshot do audytu)
cp .glx/spec_state.json .glx/spec_state.before_freeze.json

# 2. Zamroź progi (bez aktualizacji kwantyli na N commitów)
python -m glitchlab.spec.calibrate freeze --commits N --reason "drift guard"

# 3. Oznacz zakres zbierania etykiet (czy PR był OK)
git log --oneline -n N > .glx/spec_labels_scope.txt

# 4. Włącz etykietowanie w CI (tymczasowe progi statyczne)
python -m glitchlab.core.invariants --ci --freeze .glx/spec_state.json
````

**Artefakty:** `.glx/spec_state.before_freeze.json`, `.glx/spec_state.json (frozen)`, log w `analysis/logs/*`.

**Unfreeze (po diagnozie):**

```bash
python -m glitchlab.spec.calibrate unfreeze
python -m glitchlab.spec.calibrate update --window 90d --by-module
```

**Rollback:** przywrócenie snapshotu `spec_state.before_freeze.json`.

**Uwagi:** podczas freeze — CI raportuje naruszenia, ale kwantyle nie są uczone.

---

## 2) Rollback patcha naruszającego I-gates

**Kiedy:** merge spowodował naruszenie I1–I4 (np. niekomutacja Φ/Δ).

**Wejścia:** hash commita, `delta_report.json`, mozaika Δ (`.glx/mosaic_*.png`).

**Kroki:**

```bash
# 1. Zidentyfikuj commit (lub PR)
git show --name-only <SHA>

# 2. Wygeneruj diagnozę Δ
python -m glitchlab.delta.features --range <SHA>^..<SHA> --out .glx/delta_report.json
python -m glitchlab.core.invariants --commit <SHA> --report .glx/invariants_<SHA>.json

# 3. Revert (twardy)
git revert <SHA> -m 1
git push origin HEAD

# 4. Alternatywnie: hotfix branch + poprawka
git checkout -b hotfix/fix-I3-<shortsha>
# ... wprowadź poprawkę minimalną ...
pytest -q && python -m glitchlab.core.invariants --ci
git commit -am "fix: restore I3 (Φ∘Δ ≈ Δ∘Φ) after <SHA>"
git push -u origin hotfix/fix-I3-<shortsha>
```

**Artefakty:** `.glx/delta_report.json`, `.glx/invariants_<SHA>.json`.

**Rollback:** `git revert` jest sam w sobie rollbackiem; jeśli poprawka — standardowa procedura PR.

**Uwagi:** preferuj minimalny hunk; zabezpiecz testami przywracającymi komutację.

---

## 3) Reindeksacja AST/Δ i rekalkulacja mozaiki

**Kiedy:** zmiany szerokie w strukturze repo, modyfikacje w `mosaic/`, rozjazd map AST↔Mozaika.

**Wejścia:** zakres `BASE..HEAD`, polityka `analysis/policy.json`.

**Kroki:**

```bash
# 1. Reindeks AST i Δ dla zakresu
python -m analysis.ast_index --from BASE --to HEAD --out .glx/ast_index.json
python -m analysis.ast_delta  --from BASE --to HEAD --out .glx/ast_delta.json

# 2. Rekalkulacja mozaiki hybrydowej (gałąź mosaic/)
python -m mosaic.hybrid_ast_mosaic \
  --mosaic hybrid --rows 64 --cols 64 \
  --edge-thr 0.12 --kappa-ab 0.6 --phi policy \
  --policy-file analysis/policy.json \
  --repo-root . from-git-dump --base BASE --head HEAD \
  --delta 0.2 --out .glx/mosaic --strict-artifacts

# 3. Walidacja artefaktów i I-gates
test -f .glx/mosaic/mosaic_map.json
python -m glitchlab.core.invariants --ci
```

**Artefakty:** `.glx/ast_index.json`, `.glx/ast_delta.json`, `.glx/mosaic/*`.

**Rollback:** przywrócenie poprzednich artefaktów `.glx/mosaic/*` z `backup/AUDIT_*.zip`.

**Uwagi:** parametry (`rows/cols/edge-thr/kappa/Δ`) zgodnie z `docs/14_mosaic.md`.

---

## 4) Rebuild EGDB, widoki i sanity check EGQL

**Kiedy:** migracja schematów, uszkodzenie bazy, potrzeba weryfikacji widoków.

**Wejścia:** `.glx/grammar/views.sql`, `analysis/grammar/egdb_store.py`, logi.

**Kroki:**

```bash
# 1. Rebuild/weryfikacja fizycznej bazy (SQLite domyślnie)
python -m analysis.grammar.egdb_store --init --db .glx/egdb.db

# 2. Zastosuj widoki i indeksy
sqlite3 .glx/egdb.db < .glx/grammar/views.sql

# 3. Sanity check EGQL (przykłady)
python - <<'PY'
from pathlib import Path
db=".glx/egdb.db"
q="SELECT count(*) FROM events WHERE topic LIKE 'invariants.%';"
import sqlite3; c=sqlite3.connect(db); print("invariants events:", c.execute(q).fetchone())
PY
```

**Artefakty:** `.glx/egdb.db`, logi migracji.

**Rollback:** kopia pliku bazy sprzed migracji.

**Uwagi:** dla Postgresa — odpowiednik `psql -f .glx/grammar/views.sql` + migracje wersjonowane.

---

## 5) Incident: Wyciek sekretu / znalezisko SAST CRITICAL

**Kiedy:** SAST Bridge zgłasza `Secrets|CRITICAL`, Gitleaks/pip-audit CRITICAL.

**Wejścia:** `.glx/security/queue.json` (PQ), NF z `merged.sarif`.

**Kroki:**

```bash
# 1. Zidentyfikuj NF i plik
jq '.[] | select(.category=="Secrets" and .severity=="CRITICAL") | {file: .location.file, msg: .message}' .glx/security/queue.json

# 2. Natychmiast: rotacja kluczy w zewnętrznym KMS (ręczne)

# 3. Usuń z repo (bez rewrite historii na gorąco)
git rm --cached path/to/secret.file
echo "SECRET_ENV=..." >> .env.local.example  # placeholder
git commit -m "security: remove exposed secret; add placeholder"

# 4. Polityka: dodaj deny-rule (allowlist/denylist) i wzorzec w SAST Bridge
git add security/policy.yaml
git commit -m "policy: deny secret pattern"

# 5. Weryfikacja
python -m security.sast_bridge scan --out .glx/security/queue.json
```

**Artefakty:** zaktualizowany `queue.json`, polityka deny.

**Rollback:** brak — wycieku nie cofamy; ewentualny `git filter-repo` w okienku maintenance (po konsultacji).

**Uwagi:** zawsze dokumentuj rotację; nigdy nie pushuj nowych sekretów.

---

## 6) Unstuck CI: pipeline czerwony, gating na Δ

**Kiedy:** Actions padają na `invariants-check`/`SAST`, ale PR merytorycznie drobny.

**Wejścia:** log CI, `.glx/delta_report.json`.

**Kroki:**

```bash
# 1. Lokalnie odtwórz błąd
pytest -q && python -m glitchlab.core.invariants --ci
python -m glitchlab.delta.features --out .glx/delta_report.json

# 2. Sprawdź tokeny, czy „głośny” (np. ΔTESTS<0)
jq '.tokens | group_by(.kind) | map({k:.[0].kind, n:length})' .glx/delta_report.json

# 3. Działanie kontekstowe:
#   a) brak testów → dodaj stub i pokrycie
#   b) MODIFY_SIG → dodaj adapter/stub
#   c) duży ΔIMPORT → rozbij na mniejsze PR-y
```

**Artefakty:** `.glx/delta_report.json`, dodatkowe testy/adapters.

**Rollback:** rozbicie PR na mniejsze lub rebase.

**Uwagi:** nie wyłączaj bramek — zredukuj Δ do akceptowalnego „budżetu”.

---

## 7) Perf regression: wzrost czasu kroków / spadek SSIM

**Kiedy:** testy `perf` czerwone, δ-czasów > β, ΔSSIM < α.

**Wejścia:** `tests/perf/*`, wyniki HUD, `analysis/metrics.py`.

**Kroki:**

```bash
# 1. Profil krótkiej ścieżki
pytest tests/perf/test_perf_timings.py -q -k "baseline" --durations=10

# 2. Identyfikacja hot-spot (np. Perlin fallback)
python -X perf -m glitchlab.core.pipeline --preset presets/default.yaml

# 3. Szybkie poprawki:
#    - zamień pętle na vektorowe operacje
#    - cache okien (FFT), redukcja binów, unikanie kopii uint8↔float32
#    - downsample→blur→upsample dla drogich kroków

# 4. Weryfikacja metryk wizualnych
python -m glitchlab.analysis.metrics --compare A.png B.png --out .glx/compare.json
```

**Artefakty:** raporty perf (`--durations`), `.glx/compare.json`.

**Rollback:** revert regresyjnej zmiany.

**Uwagi:** utrzymuj testy *perf smoke* lekkie; pełne benchmarki tylko w nightly.

---

## 8) HUD triage: Delta Inspector & Spec Monitor

**Kiedy:** szybka diagnoza „co i gdzie się stało” w PR/commicie.

**Wejścia:** `.glx/mosaic/*`, `.glx/delta_report.json`.

**Kroki:**

1. Uruchom GUI/APP (`python -m glitchlab.gui.app`).
2. Panel **Delta Inspector**:

   * przełącz zakres commitów, zobacz histogram tokenów i heatmapę,
   * kliknij plik/token → skok do linii/sekcji.
3. Panel **Spec Monitor**:

   * sprawdź wartości α/β/Z, status freeze/drift, przyczyny naruszeń,
   * kliknij „Explain” → pokazuje wkład tokenów do score.

**Artefakty:** obrazy mozaiki, raporty Δ i I-gates.

**Rollback:** nie dotyczy.

**Uwagi:** panele są tylko nad wizualizacją — decyzje (merge/block) w CI/PR.

---

## 9) Release/hotfix procedura operacyjna

**Kiedy:** przygotowanie wydania *beta/stable* lub szybki hotfix.

**Wejścia:** `VERSION.json`, zielony CI, brak naruszeń I-gates, SAST ok.

**Kroki (skrót):**

```bash
# 1. Podbij wersję (beta/stable) i zataguj
jq '.version="1.6.0-beta.2" | .channel="beta"' VERSION.json > VERSION.tmp && mv VERSION.tmp VERSION.json
git commit -am "chore(release): 1.6.0-beta.2"
git tag v1.6.0-beta.2 && git push origin --tags

# 2. Poczekaj na workflow 'release' (artefakty, SBOM, podpisy)

# 3. Hotfix (gdy krytyk): osobna gałąź i patch minimalny
git checkout -b hotfix/1.6.1-critical
# ... fix ...
pytest -q && python -m glitchlab.core.invariants --ci
git commit -am "fix: critical hotfix"
git tag v1.6.1 && git push origin --tags
```

**Artefakty:** patrz `docs/80_release.md`.

**Rollback:** oznacz release jako *yanked*; przywróć poprzedni tag jako „latest”.

**Uwagi:** *stable* wymaga twardych bramek (mutation-score, perf, SAST=0 CRIT).

---

## 10) Disaster Recovery: odtworzenie ze snapshotu `.glx` / `backup/AUDIT_*.zip`

**Kiedy:** uszkodzenie artefaktów `.glx/*`, potrzeba reprodukcji stanu.

**Wejścia:** `backup/AUDIT_*.zip`.

**Kroki:**

```bash
# 1. Wylistuj dostępne snapshoty
ls -1 backup/AUDIT_*.zip | tail

# 2. Odtwórz do katalogu tymczasowego
mkdir -p .glx_restore && unzip backup/AUDIT_20251004-231343.zip -d .glx_restore

# 3. Zweryfikuj integralność (hashy plików)
python - <<'PY'
import hashlib, pathlib
for p in pathlib.Path(".glx_restore").rglob("*"):
    if p.is_file():
        print(hashlib.sha256(p.read_bytes()).hexdigest(), p)
PY

# 4. Podmień brakujące artefakty
rsync -av .glx_restore/.glx/ .glx/
```

**Artefakty:** przywrócone `.glx/*`.

**Rollback:** zachowaj kopię `.glx` sprzed przywrócenia.

**Uwagi:** snapshot zawiera także `GLX_AUDIT_META.json` — przyda się w audycie.

---

## 11) Reindex presetów/filtrów i kwarantanna pluginów

**Kiedy:** dodano/zmieniono filtry lub preset; podejrzenie niestabilnego pluginu.

**Wejścia:** `filters/*.py`, `presets/*.yaml`.

**Kroki:**

```bash
# 1. Reindex presetów i filtrów (GUI ładuje dynamicznie, tu weryfikacja)
python -m gui.preset_manager --index --out .glx/presets_index.json
python -m gui.panel_loader --validate filters/ --qos sandbox

# 2. Kwarantanna pluginu (oddzielny proces + whitelist)
python -m gui.services.pipeline_runner --sandbox --plugin filters/suspicious.py
```

**Artefakty:** `.glx/presets_index.json`, log sandbox.

**Rollback:** usuń z kwarantanny po weryfikacji.

**Uwagi:** polityki sandbox patrz `docs/60_security.md`.

---

## 12) Playbook SRE: „Δ spike > Z” (nagły wzrost energii zmian)

**Kiedy:** energia ΔZ dla PR/commitu przekracza Z (alarm w HUD/CI).

**Wejścia:** `delta_report.json`, mozaika Δ, `Spec Monitor`.

**Kroki (triage → containment → recovery):**

1. **Triage**

   * Otwórz **Delta Inspector** → sprawdź top-tokeny (np. `RENAME`, `ΔIMPORT`, `EXTRACT_FN`, `ΔTESTS`).
   * Ustal, czy spike jest *strukturalny* (rename/refactor) czy *semantyczny* (zmiana zachowania).
2. **Containment**

   * Jeśli *strukturalny*: rozbij PR na paczki per moduł/panel; dodaj adaptery/stuby.
   * Jeśli *semantyczny*: zamroź progi (freeze), włącz pełne testy e2e.
3. **Recovery**

   * Dla `MODIFY_SIG`: generator adapterów/stub-tests → zredukuj ΔTESTS do ≥ 0.
   * Dla `ΔIMPORT↑`: usuń importy krzyżowe, wprowadź porty lub rejestry (patrz `docs/41_pipelines.md`).
   * Przebuduj mozaikę i przeliczniki Δ (Playbook #3), waliduj I-gates.

**Artefakty:** zaktualizowany `delta_report.json`, mozaiki, log freeze/unfreeze.

**Rollback:** jeśli nie da się zbić ΔZ < β — rozbij PR lub revert.

**Uwagi:** nie podnoś progów ad hoc; używaj freeze + etykiet, potem *unfreeze z danymi*.

---

## Appendix A — Przydatne fragmenty EGQL/SQL

**Najgorętsze pliki (top energia ΔZ) w ostatnich 20 commitach:**

```sql
SELECT file, SUM(delta_z) AS dz
FROM vw_commit_deltas
WHERE commit_ts > strftime('%s','now','-7 days')
GROUP BY file ORDER BY dz DESC LIMIT 20;
```

**Korelacja „naruszenie I3” ↔ „MODIFY_SIG”:**

```sql
SELECT COUNT(*) FROM vw_invariants_violations v
JOIN vw_delta_tokens t ON v.commit_sha=t.commit_sha
WHERE v.kind='I3' AND t.kind='MODIFY_SIG';
```

---

## Appendix B — Szablony komunikatów (PR/incident)

**PR blokowany przez I-gates:**

> This PR exceeds **β** (score={{score}}) with tokens: {{top_tokens}}.
> Violations: {{I*}}. See artifacts: `.glx/delta_report.json`, `.glx/mosaic/*`.
> Action: reduce Δ (split PR, add tests/adapters) or provide rationale.

**Incident SAST CRITICAL:**

> SAST flagged **{{rule}}** in `{{path}}:{{line}}`. Secret rotation required.
> Removed artifact from repo, deny policy added. See `.glx/security/queue.json`.

---

## Appendix C — Minimalne check-listy

**Before Merge (stable):**

* [ ] Tests (unit/e2e/perf) ✔
* [ ] I-gates (I1–I4) ✔
* [ ] Δ-fingerprint ≤ β, brak ΔTESTS<0 ✔
* [ ] SAST: 0× CRITICAL ✔
* [ ] HUD review (heatmap + tokens) ✔

**After Incident:**

* [ ] Zapisz czasy i identyfikatory commitów
* [ ] Zabezpiecz artefakty `.glx/*`
* [ ] Wypełnij raport post-mortem (przyczyna, działania, lekcje)

---

**Koniec dokumentu.**

```
::contentReference[oaicite:0]{index=0}
```
