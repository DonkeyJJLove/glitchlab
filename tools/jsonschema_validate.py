import json, sys
from pathlib import Path
from jsonschema import Draft7Validator

SCHEMAS = [
    'spec/schemas/changed_file.json',
    'spec/schemas/node_impact.json',
    'spec/schemas/refactor_action.json',
]

def main():
    root = Path('')
    ok = True
    for s in SCHEMAS:
        p = root / s
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding='utf-8'))
        Draft7Validator.check_schema(data)
        print(f'[OK] {s}')
    # Optional: validate registry if present
    reg = root / '.glx/schemas/registry.json'
    if reg.exists():
        try:
            json.loads(reg.read_text(encoding='utf-8'))
            print('[OK] .glx/schemas/registry.json is valid JSON')
        except Exception as e:
            print('[ERR] registry.json:', e); ok=False
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
