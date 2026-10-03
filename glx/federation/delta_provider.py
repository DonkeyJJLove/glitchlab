"""Read-only headless GlitchLab delta provider for federation use."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from glx.tools import delta_fingerprint
from glx.tools.invariants_check import classify_by_thresholds, compute_score_from_report

GIT = re.compile(r"^[0-9a-f]{40}$")


class DeltaProviderError(ValueError):
    pass


def _git(repo: Path, *args: str) -> bytes:
    cp = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if cp.returncode:
        raise DeltaProviderError(cp.stderr.decode("utf-8", "replace")[-1000:])
    return cp.stdout


def _resolve(repo: Path, rev: str) -> str:
    out = _git(repo, "rev-parse", "--verify", rev).decode().strip()
    if GIT.fullmatch(out) is None:
        raise DeltaProviderError("git identity")
    return out


@dataclass(frozen=True)
class DeltaObservation:
    schema: str
    repository_ref: str
    base_commit: str
    head_commit: str
    changed_files: tuple[str, ...]
    diff_sha256: str
    delta_histogram: tuple[tuple[str, int], ...]
    delta_fingerprint: str
    invariant_score: float
    invariant_block: bool
    provider_source_ref: str
    process_semantics_ref: str
    process_semantics_digest: str
    authority_effect: str = "NONE"
    mutation_effect: str = "NONE"
    observation_digest: str = ""

    def payload(self):
        d = asdict(self)
        d.pop("observation_digest", None)
        return d

    def compute_digest(self):
        raw = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return sha256(b"LION/GLITCHLAB-DELTA-OBSERVATION/1\0" + raw).hexdigest()

    def validate(self):
        if self.schema != "lion.glitchlab-delta-observation/v1":
            raise DeltaProviderError("schema")
        if (
            not self.repository_ref
            or GIT.fullmatch(self.base_commit) is None
            or GIT.fullmatch(self.head_commit) is None
        ):
            raise DeltaProviderError("identity")
        if not isinstance(self.changed_files, tuple) or self.changed_files != tuple(
            sorted(set(self.changed_files))
        ):
            raise DeltaProviderError("changed_files")
        if not isinstance(self.process_semantics_ref, str) or not self.process_semantics_ref:
            raise DeltaProviderError("process_semantics_ref")
        if (
            not isinstance(self.process_semantics_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.process_semantics_digest) is None
        ):
            raise DeltaProviderError("process_semantics_digest")
        if self.authority_effect != "NONE" or self.mutation_effect != "NONE":
            raise DeltaProviderError("effects")
        if self.observation_digest != self.compute_digest():
            raise DeltaProviderError("digest")
        return self


def observe_git_range(
    repo_root: Path,
    base: str,
    head: str,
    *,
    repository_ref: str,
    provider_source_ref: str,
    process_semantics_ref: str,
    process_semantics_digest: str,
) -> DeltaObservation:
    repo = Path(repo_root).resolve()
    if not (repo / ".git").exists():
        raise DeltaProviderError("repo root")
    b = _resolve(repo, base)
    h = _resolve(repo, head)
    if b == h:
        raise DeltaProviderError("empty commit range")
    changed = tuple(
        sorted(
            x
            for x in _git(repo, "diff", "--name-only", b + ".." + h)
            .decode("utf-8", "strict")
            .splitlines()
            if x
        )
    )
    diff = _git(repo, "diff", "--binary", b + ".." + h)
    # delta_fingerprint.run is read-only; bind its repo-root resolver to this exact root for the call.
    old = delta_fingerprint._guess_repo_root
    try:
        delta_fingerprint._guess_repo_root = lambda: repo
        report = delta_fingerprint.run(b + ".." + h)
    finally:
        delta_fingerprint._guess_repo_root = old
    score = compute_score_from_report(report)
    block = classify_by_thresholds(score)
    value = DeltaObservation(
        "lion.glitchlab-delta-observation/v1",
        repository_ref,
        b,
        h,
        changed,
        sha256(diff).hexdigest(),
        tuple(sorted((str(k), int(v)) for k, v in (report.get("hist") or {}).items())),
        str(report.get("hash") or ""),
        float(score),
        bool(block),
        provider_source_ref,
        process_semantics_ref,
        process_semantics_digest,
    )
    value = DeltaObservation(**{**asdict(value), "observation_digest": value.compute_digest()})
    return value.validate()
