"""Enforce the publication boundary across the whole repository history.

This repository does not redistribute ChartQA and does not carry model weights
or private planning documents. `scripts/verify_claims.py` checks the current
tree; this checks every path and every blob that has ever been committed, so a
reintroduction cannot hide in an earlier commit.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Paths that must never appear in any commit.
FORBIDDEN_PATHS = (
    (re.compile(r"assets/eval/cases/"), "ChartQA-derived case images and metadata"),
    (
        re.compile(r"assets/eval/predictions_"),
        "raw ChartQA predictions with queries and golds",
    ),
    (re.compile(r"assets/eval_quant/preds_"), "raw ChartQA predictions with golds"),
    (re.compile(r"space_static/assets/case_\d+\.png$"), "ChartQA-derived chart image"),
    (re.compile(r"\.env$"), "environment file"),
    (re.compile(r"(?i)\.(safetensors|gguf|bin|pt|pth|ckpt)$"), "model weight file"),
)

# Content that must never appear in any blob.
#
# Each needle is assembled from fragments so this scanner does not flag itself.
# That matters more here than in verify_claims.py: this check walks every blob
# ever committed, so once a literal copy of the needle is in history it would
# fail the build forever. Keep the `+` concatenation.
FORBIDDEN_CONTENT = (
    (b'"quer' + b'ies"', "raw ChartQA query list"),
    (b'"go' + b'lds"', "raw ChartQA gold-label list"),
    (b"smallest value on the " + b"blue bar", "verbatim ChartQA query text"),
    (b"expected answer `" + b"96`", "ChartQA case answer"),
    (b"8 \xc3\x97 12 = " + b"96", "ChartQA case answer and calculation"),
    (b"one-way `query_" + b"sha256`", "hash anonymity claim"),
    (b"kuotu" + b"nyh", "superseded private identity"),
)

# A short former test identity needs token boundaries. A raw substring search
# would mistake the same hexadecimal sequence inside legitimate SHA-256 values
# and package hashes for an account identifier.
FORBIDDEN_CONTENT_PATTERNS = (
    (
        re.compile(rb"(?<![a-z0-9])a" + rb"123(?![a-z0-9])", re.IGNORECASE),
        "superseded private identity",
    ),
)

EXPECTED_IDENTITY = "kuotunyu <61350295+kuotunyu@users.noreply.github.com>"

MAX_TRACKED_BYTES = 2_000_000  # nothing in this repo should approach a weight file


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        sys.exit(2)
    return result.stdout


def main() -> int:
    findings: list[str] = []

    ever = {
        p
        for p in git("log", "--all", "--pretty=format:", "--name-only").splitlines()
        if p
    }
    current = {p for p in git("ls-files").splitlines() if p}
    print(f"checking {len(current)} tracked paths and {len(ever)} paths ever committed")

    for pattern, why in FORBIDDEN_PATHS:
        hits = sorted(p for p in ever | current if pattern.search(p))
        for hit in hits:
            findings.append(f"forbidden path `{hit}` ({why})")

    listing = git(
        "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectname)"
    )
    blobs = [
        line.split()[1] for line in listing.splitlines() if line.startswith("blob")
    ]
    for oid in blobs:
        data = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "blob", oid], capture_output=True
        ).stdout
        for needle, why in FORBIDDEN_CONTENT:
            if needle.lower() in data.lower():
                findings.append(f"forbidden content in blob {oid[:12]} ({why})")
        for pattern, why in FORBIDDEN_CONTENT_PATTERNS:
            if pattern.search(data):
                findings.append(f"forbidden content in blob {oid[:12]} ({why})")
    print(f"scanned {len(blobs)} history blobs")

    for rel in current:
        path = ROOT / rel
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            findings.append(
                f"tracked file `{rel}` is {path.stat().st_size} bytes "
                f"(limit {MAX_TRACKED_BYTES}) — weights and datasets belong on the Hub"
            )

    ALLOWED_BOT_IDENTITIES = {
        "GitHub <noreply@github.com>",
        "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
        "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>",
    }
    identities = {
        line
        for line in git("log", "--all", "--format=%an <%ae>").splitlines()
        + git("log", "--all", "--format=%cn <%ce>").splitlines()
        if line
    }
    human_identities = identities - ALLOWED_BOT_IDENTITIES
    if human_identities != {EXPECTED_IDENTITY}:
        findings.append(f"unexpected commit identities: {sorted(human_identities)}")

    remotes = git("remote").split()
    print(f"identities: {sorted(identities)}")
    print(f"remotes: {remotes or '(none)'}")

    if findings:
        print(f"\nFAILED: {len(findings)} boundary violation(s)")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("PASS: publication boundary intact across the full history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
