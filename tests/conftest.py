# glitchlab/tests/conftest.py
import os, sys, re
from pathlib import Path

def _load_env_file(env_path: Path):
    if not env_path.exists():
        return {}
    txt = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    env = {}
    for line in txt:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        # zdejmij otaczające cudzysłowy, jeśli są
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        env[k] = v
    return env

# 1) Najpierw spróbuj z istniejących zmiennych środowiskowych
glx_root = os.getenv("GLX_ROOT")

# 2) Jeśli brak → wczytaj .env z katalogu projektu (2 poziomy wyżej od /glitchlab/tests/)
if not glx_root:
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"
    env_map = _load_env_file(env_file)
    glx_root = env_map.get("GLX_ROOT")

# 3) Dołóż GLX_ROOT na początek sys.path, żeby `import glitchlab` działał
if glx_root:
    p = str(Path(glx_root).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
