#!/usr/bin/env python
# glitchlab/analysis/reporting.py
# -*- coding: utf-8 -*-
# Python 3.9+

from __future__ import annotations

import re
import json
import os
import sys
import subprocess
import zipfile
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── GLX_RUN: flags & kolejność ───────────────────────────────────────────────
FLAGS = {"A": 0x1, "M": 0x2, "E": 0x4, "Z": 0x8}
ORDER_CANON = ["A", "M", "E", "Z"]  # fallback dla postaci liczbowych


def _strip_inline_comment(s: str) -> str:
    """Usuwa #... tylko gdy nie jesteśmy w cudzysłowach."""
    s = s or ""
    in_s = in_d = False
    out = []
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == '#' and not in_s and not in_d:
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_glx_run(s: str) -> Tuple[int, List[str]]:
    """
    Obsługuje: 'A+M|Z', 'amez', 'MEAZ', '0xF', '15'.
    Zwraca: (mask, order) — order to KOLEJNOŚĆ kroków z wejścia (bez duplikatów).
    Dla postaci liczbowych — kolejność kanoniczna ORDER_CANON.
    """
    s0 = _strip_inline_comment((s or "").strip())
    if not s0:
        _fail("GLX_RUN jest pusty. Podaj np. 'A', 'A+M', 'MEAZ', '0xD' lub '13'.")

    up = s0.upper().replace(" ", "")

    # 1) Hex / decimal
    try:
        if up.startswith("0X"):
            mask = int(up, 16)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                _fail("GLX_RUN=0x0 nie zawiera żadnych trybów.")
            return mask, order
        if up.isdigit():
            mask = int(up, 10)
            order = [k for k in ORDER_CANON if FLAGS[k] & mask]
            if not mask:
                _fail("GLX_RUN=0 nie zawiera żadnych trybów.")
            return mask, order
    except ValueError:
        pass

    # 2) Litery z separatorami
    if re.findall(r"[^AMEZ\s,+;/|.\-]", up):
        _fail("GLX_RUN zawiera niedozwolone znaki (dozwolone A,M,E,Z i separatory).")

    letters = re.findall(r"[AMEZ]", up)
    if not letters:
        _fail("GLX_RUN nie zawiera liter z zestawu [A M E Z].")

    order, seen = [], set()
    for ch in letters:
        if ch not in seen:
            seen.add(ch)
            order.append(ch)

    mask = 0
    for ch in order:
        mask |= FLAGS[ch]
    return mask, order


def _validate_mask(mask: int) -> None:
    """Minimalna logika zdrowego rozsądku."""
    # Z (audit ZIP) wymaga przynajmniej jednego z A/M (bo musi mieć co archiwizować).
    if (mask & FLAGS["Z"]) and not (mask & (FLAGS["A"] | FLAGS["M"])):
        _fail("GLX_RUN zawiera Z bez A/M — ZIP audytu wymaga artefaktów z Autonomii lub Mozaiki.")


# ── Mini-loader .env ─────────────────────────────────────────────────────────
def _clean_value(v: str) -> str:
    """Trym, usuń cudzysłowy i przypadkowe dwukropki na końcu."""
    v = _strip_inline_comment(v or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        v = v[1:-1]
    v = v.rstrip(":").strip()
    return v


def _parse_env_file(p: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = _clean_value(v)
    return out


def _load_env(repo_root: Path) -> Dict[str, str]:
    e1 = _parse_env_file(repo_root / ".env")
    e2 = _parse_env_file(repo_root / ".env.local")  # override lokalny
    env = {**e1, **e2}
    if not env:
        _fail(f"Brak .env/.env.local w {repo_root} (lub puste).")
    return env


def _expand_vars_and_user(s: str) -> str:
    # Rozszerza $VAR / %VAR% i ~
    return os.path.expanduser(os.path.expandvars(s or ""))


def _resolve_path(val: str, base: Path) -> Path:
    """
    Normalizuj ścieżkę .env:
      - rozwiń zmienne i ~,
      - absolutna → resolve(),
      - względna → base / val,
      - pusta lub '.' → base.
    """
    v = _expand_vars_and_user(_clean_value(val))
    if not v or v == ".":
        return base.resolve()
    p = Path(v)
    return (p if p.is_absolute() else (base / p)).resolve()


def _norm_paths_in_env(env: Dict[str, str], repo_root: Path) -> Dict[str, str]:
    """
    Zasada: ścieżki względne liczymy względem GLX_ROOT.
    Najpierw normalizujemy GLX_ROOT względem repo_root, potem resztę względem GLX_ROOT.
    """
    glx_root = _resolve_path(env.get("GLX_ROOT", "."), repo_root)
    env["GLX_ROOT"] = str(glx_root)
    for key in ("GLX_OUT", "GLX_AUTONOMY_OUT", "GLX_POLICY"):
        if key in env:
            env[key] = str(_resolve_path(env.get(key, ""), glx_root))
    return env


# ── Log i błędy ──────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    print(f"[GLX][post-commit] {msg}")


def _fail(msg: str, code: int = 2) -> None:
    print(f"[GLX][ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# ── GIT utils ────────────────────────────────────────────────────────────────
def _git(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd))


def _git_top(cwd: Path) -> Optional[Path]:
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(cwd), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except Exception:
        pass
    return None


def _rev_parse(ref: str, cwd: Path) -> str:
    r = _git(["rev-parse", ref], cwd=cwd)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else ref


def _detect_base(cwd: Path, state_p: Path) -> str:
    # 1) .glx/state.json.base_sha
    if state_p.exists():
        try:
            data = json.loads(state_p.read_text(encoding="utf-8"))
            bs = str(data.get("base_sha") or "").strip()
            if bs:
                return bs
        except Exception as e:
            _log(f"state.json read warning: {e}")
    # 2) merge-base z origin/master lub origin/main
    for br in ("origin/master", "origin/main"):
        r = _git(["merge-base", "HEAD", br], cwd=cwd)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    # 3) HEAD~1
    r = _git(["rev-parse", "HEAD~1"], cwd=cwd)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return "HEAD~1"


# ── Walidacja i rzutowanie typów ─────────────────────────────────────────────
def _req(env: Dict[str, str], key: str) -> str:
    if key not in env or str(env[key]).strip() == "":
        _fail(f"Wymagany klucz .env brakujący: {key}")
    return str(env[key]).strip()


def _as_int(env: Dict[str, str], key: str, lo: int, hi: int) -> int:
    v = int(_req(env, key))
    if not (lo <= v <= hi):
        _fail(f"{key}={v} poza zakresem [{lo},{hi}]")
    return v


def _as_float01(env: Dict[str, str], key: str) -> float:
    v = float(_req(env, key))
    if not (0.0 <= v <= 1.0):
        _fail(f"{key}={v} poza zakresem [0,1]")
    return v


# ── Snapshot env (z redakcją sekretów) ───────────────────────────────────────
_SECRET_HINTS = ("PASS", "PASSWORD", "SECRET", "TOKEN", "KEY")


def _redact_env(d: Dict[str, str]) -> Dict[str, str]:
    red: Dict[str, str] = {}
    for k, v in d.items():
        if any(h in k.upper() for h in _SECRET_HINTS):
            red[k] = "******"
        else:
            red[k] = v
    return red


# ── Autodetekcja modułów / fallback do plików ────────────────────────────────
def _module_exists(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _module_cmd_from_candidates(candidates: List[str], root: Path) -> List[str]:
    """
    Zwraca polecenie do uruchomienia:
      - preferuj uruchomienie modułu przez `-m`, jeśli spec istnieje,
      - w przeciwnym razie spróbuj bezpośrednio pliku .py w repo (fallback).
    """
    # 1) moduł importowalny?
    for mod in candidates:
        if _module_exists(mod):
            return [sys.executable, "-m", mod]

    # 2) fallback do plików .py
    for mod in candidates:
        rel = Path(*mod.split("."))
        for cand in (root / f"{rel}.py", root / rel / "__main__.py"):
            if cand.exists():
                return [sys.executable, str(cand)]

    _fail("Nie znaleziono modułu ani skryptu dla: " + " | ".join(candidates))
    return []  # unreachable


# ── Kroki ────────────────────────────────────────────────────────────────────

def _child_env(env: Dict[str, str]) -> Dict[str, str]:
    e = os.environ.copy()
    for k, v in env.items():
        e[k] = str(v)
    return e


def _run_gateway(env: Dict[str, str], repo_root: Path) -> None:
    pkg = _req(env, "GLX_PKG")
    out_dir = repo_root / _req(env, "GLX_AUTONOMY_OUT")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _detect_base(repo_root, repo_root / ".glx" / "state.json")
    head = _rev_parse("HEAD", repo_root)

    cmd = _module_cmd_from_candidates(
        [f"{pkg}.analysis.autonomy.gateway", f"{pkg}.gui.analysis.autonomy.gateway", "analysis.autonomy.gateway"],
        repo_root
    ) + ["build", "--out", str(out_dir), "--base", base, "--head", head]

    _log("gateway: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        _fail(f"gateway zakończył się kodem {r.returncode}")

    # wymagane artefakty (pack.json; w okresie przejściowym akceptuj też packz.json)
    must = {"pack.json", "pack.md", "prompt.json"}
    present = {p.name for p in out_dir.glob("*")}
    if "pack.json" not in present and "packz.json" in present:
        # kompatybilność wsteczna: traktuj packz.json jak pack.json
        try:
            (out_dir / "pack.json").write_text((out_dir / "packz.json").read_text(encoding="utf-8"), encoding="utf-8")
            present.add("pack.json")
        except Exception as e:
            _fail(f"Nie udało się zmapować packz.json→pack.json: {e}")

    missing = [fn for fn in must if fn not in present]
    if missing:
        _fail("Brak wymaganych artefaktów autonomii: " + ", ".join(missing))

    # zapisz snapshot env
    snap = _redact_env(env)
    (out_dir / "env.json").write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_mosaic(env: Dict[str, str], repo_root: Path) -> None:
    pkg = _req(env, "GLX_PKG")
    mosaic = _req(env, "GLX_MOSAIC")
    rows = _as_int(env, "GLX_ROWS", 1, 512)
    cols = _as_int(env, "GLX_COLS", 1, 512)
    edge_thr = _as_float01(env, "GLX_EDGE_THR")
    psi = _as_float01(env, "GLX_DELTA")  # Ψ
    kappa = _as_float01(env, "GLX_KAPPA")  # κ
    phi = _req(env, "GLX_PHI").lower()
    policy = env.get("GLX_POLICY", "analysis/policy.json")

    base = _detect_base(repo_root, repo_root / ".glx" / "state.json")

    # Autodetekcja modułu: preferuj <pkg>.mosaic..., potem <pkg>.gui.mosaic..., itd.
    mosaic_candidates = [
        f"{pkg}.mosaic.hybrid_ast_mosaic",
        f"{pkg}.gui.mosaic.hybrid_ast_mosaic",
        "mosaic.hybrid_ast_mosaic",
        "gui.mosaic.hybrid_ast_mosaic",
    ]
    cmd: List[str] = _module_cmd_from_candidates(mosaic_candidates, repo_root) + [
        "--mosaic", mosaic, "--rows", str(rows), "--cols", str(cols),
        "--edge-thr", str(edge_thr), "--kappa-ab", str(kappa),
        "--phi", phi
    ]
    if phi == "policy":
        pol_p = (repo_root / policy) if not Path(policy).is_absolute() else Path(policy)
        if not pol_p.exists():
            _fail(f"GLX_PHI=policy ale brak pliku polityki: {pol_p}")
        cmd += ["--policy-file", str(pol_p)]

    cmd += [
        "from-git-dump", "--base", base, "--head", "HEAD",
        "--delta", str(psi),
        "--out", str(repo_root / _req(env, "GLX_OUT")),
        "--strict-artifacts",  # ✨ DODANE
    ]

    _log("mosaic: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(repo_root))
    if r.returncode != 0:
        _fail(f"mozaika zakończyła się kodem {r.returncode}")

    # walidacja artefaktów mozaiki
    out_dir = repo_root / _req(env, "GLX_OUT")
    candidates = list(out_dir.rglob("report.json")) + list(out_dir.rglob("mosaic_map.json")) + list(
        out_dir.rglob("summary.md"))
    if not candidates:
        listing = "\n".join(f" - {p.relative_to(repo_root)}" for p in sorted(out_dir.rglob("*")) if p.is_file())
        _fail("Brak artefaktów mozaiki w " + str(out_dir) + "\nPliki znalezione:\n" + (listing or "(pusto)"))


def _send_mail(env: Dict[str, str]) -> None:
    # Tylko kontrola wymagań; sama implementacja maila jest po stronie modułu.
    for k in ("GLX_SMTP_HOST", "GLX_SMTP_PORT", "GLX_SMTP_USER", "GLX_SMTP_PASS", "GLX_MAIL_TO"):
        _req(env, k)

    try:
        try:
            from scripts.send_patch import main as glx_main
        except Exception:
            from scripts.send_patch import main as glx_main  # type: ignore
    except Exception as e:
        _fail(f"Import modułu maila nieudany: {e}")

    try:
        glx_main()
    except SystemExit as e:
        if int(e.code or 0) != 0:
            _fail(f"send_patch zakończył się kodem {e.code}")
    except Exception as e:
        _fail(f"send_patch wyjątek: {e}")


# ── ZIP audytu (deterministyczny + filtry) ───────────────────────────────────
def _should_zip(p: Path) -> bool:
    rel = str(p).replace("\\", "/")
    if "/.git/" in rel or "/__pycache__/" in rel:
        return False
    return p.is_file()


def _writestr_deterministic(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    # Stały timestamp dla stabilnych hashy ZIP
    zi = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(zi, data)


def _make_zip(env: Dict[str, str], repo_root: Path) -> None:
    out_dir = repo_root / _req(env, "GLX_OUT")
    auto_dir = repo_root / _req(env, "GLX_AUTONOMY_OUT")
    backups = repo_root / ".glx" / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_path = backups / f"AUDIT_{ts}.zip"

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for root in (auto_dir, out_dir):
            if root.exists():
                for p in root.rglob("*"):
                    if _should_zip(p):
                        arc = str(p.relative_to(repo_root)).replace("\\", "/")
                        with p.open("rb") as f:
                            _writestr_deterministic(zf, arc, f.read())
        meta = {
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "glx_run": _req(env, "GLX_RUN"),
            "project": env.get("GLX_PROJECT", "?"),
            "branch": env.get("GLX_BRANCH", "?"),
            "ver": env.get("GLX_VER", "?"),
        }
        _writestr_deterministic(zf, "GLX_AUDIT_META.json",
                                json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
    _log(f"audit zip: {zip_path.name}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> int:
    # 0) Wyznacz repo_root: preferuj git-top, potem heurystyka po ścieżce pliku
    here = Path(__file__).resolve()
    repo_root = _git_top(here) or here.parents[2]

    # 1) Wymagamy .env / .env.local w repo_root
    if not (repo_root / ".env").exists() and not (repo_root / ".env.local").exists():
        _fail(f"Nie znaleziono .env ani .env.local w {repo_root}")

    # 2) Załaduj .env i znormalizuj ścieżki
    env = _load_env(repo_root)
    env = _norm_paths_in_env(env, repo_root)

    # 3) GLX_RUN jako mask + kolejność + sanity kombinacji
    mask, order = _parse_glx_run(env.get("GLX_RUN", ""))
    _validate_mask(mask)

    # 4) Walidacje bazowe (już po normalizacji)
    for k in ("GLX_ROOT", "GLX_PKG", "GLX_OUT", "GLX_AUTONOMY_OUT"):
        _req(env, k)

    # 5) PYTHONPATH – deterministycznie (obsłuż przypadek GLX_ROOT=.../glitchlab)
    glx_root = Path(env["GLX_ROOT"])
    if not glx_root.exists():
        _fail(f"GLX_ROOT wskazuje na nieistniejący katalog: {glx_root}")

    pkg = _req(env, "GLX_PKG")  # np. 'glitchlab'
    # jeśli GLX_ROOT zawiera folder pakietu -> import_root = GLX_ROOT
    # jeśli GLX_ROOT to sam folder pakietu -> import_root = parent
    if (glx_root / pkg).is_dir():
        import_root = glx_root
    elif glx_root.name == pkg and glx_root.parent.exists():
        import_root = glx_root.parent
    else:
        import_root = glx_root

    for p in (str(import_root), str(glx_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    _log(f"import_root={import_root}  glx_root={glx_root}")

    # 6) CWD = GLX_ROOT (zgodnie z polityką ścieżek względnych)
    os.chdir(str(glx_root))

    # 7) Dispatcher po KOLEJNOŚCI z wejścia
    DISPATCH = {
        "A": lambda: (_log("run:A"), _run_gateway(env, glx_root)),
        "M": lambda: (_log("run:M"), _run_mosaic(env, glx_root)),
        "E": lambda: (_log("run:E"), _send_mail(env)),
        "Z": lambda: (_log("run:Z"), _make_zip(env, glx_root)),
    }
    for ch in order:
        DISPATCH[ch]()  # kolejność rzeczywista = kolejność z GLX_RUN

    _log(f"OK (mask=0x{mask:X}, order={''.join(order)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
