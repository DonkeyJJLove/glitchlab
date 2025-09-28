#!/usr/bin/env python3
"""
pycharm_raport_praser.py
Parser raportów SARIF (np. z PyCharm inspection) do prostego raportu podatności.

Wejście: plik SARIF JSON (np. inspection.sarif.json)
Wyjście: raport tekstowy w formacie:
    [RULE] Incorrect call arguments (PyArgumentList)
    - Plik: glitchlab/gui/app.py
    - Linia: 152
    - Kolumna: 31-43
    - Poziom: warning
    - Opis: Unexpected argument
    - Fragment: menu=menubar
"""

import json
from pathlib import Path

# 🔹 tutaj ustaw swoją ścieżkę do pliku SARIF
INPUT_FILE = r"inspection.sarif.json"


def parse_sarif(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        sarif = json.load(f)

    runs = sarif.get("runs", [])
    for run in runs:
        tool = run.get("tool", {}).get("driver", {})
        tool_name = tool.get("name", "UnknownTool")
        tool_version = tool.get("version", "N/A")

        print(f"=== Raport: {tool_name} v{tool_version} ===")
        print()

        for result in run.get("results", []):
            rule_id = result.get("ruleId", "UnknownRule")
            message = result.get("message", {}).get("text", "")
            level = result.get("level", "info")
            kind = result.get("kind", "")
            rule_name = None

            # dopasowanie nazwy reguły
            for r in tool.get("rules", []):
                if r.get("id") == rule_id:
                    rule_name = r.get("name")
                    break

            print(f"[RULE] {rule_name or 'No name'} ({rule_id})")
            print(f"- Poziom: {level} ({kind})")
            print(f"- Opis: {message}")

            for loc in result.get("locations", []):
                phys = loc.get("physicalLocation", {})
                artifact = phys.get("artifactLocation", {})
                region = phys.get("region", {})

                file_uri = artifact.get("uri", "UnknownFile")
                # oczyszczanie ścieżki
                file_path = Path(file_uri.replace("file://", "").replace("..\\ile://", ""))

                start_line = region.get("startLine", "?")
                start_col = region.get("startColumn", "?")
                end_col = region.get("endColumn", "?")
                snippet = region.get("snippet", {}).get("text", "")

                print(f"- Plik: {file_path}")
                print(f"- Linia: {start_line}")
                print(f"- Kolumna: {start_col}-{end_col}")
                if snippet:
                    print(f"- Fragment: {snippet}")

            print("-" * 40)


if __name__ == "__main__":
    parse_sarif(INPUT_FILE)
