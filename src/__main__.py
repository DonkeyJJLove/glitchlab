# glitchlab/__main__.py
from __future__ import annotations
import json
from pathlib import Path
import typer
from typing import Optional
from glitchlab.delta.tokens import extract_from_git, extract_from_files, extract_from_sources
from glitchlab.delta.fingerprint import fingerprint_from_tokens
from glitchlab.delta.features import features_from_tokens

app = typer.Typer(help="GlitchLab CLI")

@app.command("delta-from-git")
def delta_from_git(
    repo: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True),
    base: str = typer.Option("HEAD~1", help="Base revision"),
    head: str = typer.Option("HEAD", help="Head revision"),
    out: Optional[Path] = typer.Option(None, help="Write JSON report to this path"),
):
    tokens = extract_from_git(repo, base, head)
    fp = fingerprint_from_tokens(tokens)
    feats = features_from_tokens(tokens)
    report = {
        "repo": str(repo),
        "base": base,
        "head": head,
        "tokens": [t.__dict__ for t in tokens],
        "fingerprint": {"histogram": fp.histogram, "weighted_sum": fp.weighted_sum, "digest": fp.digest},
        "features": feats,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        typer.echo(json.dumps(report, indent=2))

@app.command("delta-from-files")
def delta_from_files(
    prev: Path = typer.Argument(..., exists=False),
    curr: Path = typer.Argument(..., exists=False),
):
    tokens = extract_from_files(prev if prev.exists() else None, curr if curr.exists() else None)
    fp = fingerprint_from_tokens(tokens)
    typer.echo(json.dumps({"tokens":[t.__dict__ for t in tokens], "fingerprint":{"histogram": fp.histogram, "weighted_sum": fp.weighted_sum, "digest": fp.digest}}, indent=2))

if __name__ == "__main__":
    app()
