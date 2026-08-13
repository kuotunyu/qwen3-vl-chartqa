"""Scan the working tree and the entire git object database for secrets.

`scripts/verify_claims.py` scans tracked files; this goes further and walks every
blob ever committed, so a credential removed in a later commit still fails the
build. Exits non-zero on the first category with findings.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS = (
    (re.compile(rb"hf_[A-Za-z0-9]{34,}"), "Hugging Face user token"),
    (re.compile(rb"ghp_[A-Za-z0-9]{36,}"), "GitHub personal access token"),
    (re.compile(rb"gho_[A-Za-z0-9]{36,}"), "GitHub OAuth token"),
    (re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"), "GitHub fine-grained token"),
    (re.compile(rb"sk-[A-Za-z0-9]{32,}"), "OpenAI-style secret key"),
    (re.compile(rb"sk-ant-[A-Za-z0-9\-_]{20,}"), "Anthropic API key"),
    (re.compile(rb"AIza[A-Za-z0-9_\-]{35}"), "Google API key"),
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (
        re.compile(
            rb"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{24,}['\"]"
        ),
        "hard-coded credential assignment",
    ),
)

# Files that legitimately contain long random-looking strings.
ALLOWED_SUFFIXES = (".lock",)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        sys.exit(2)
    return result.stdout


def scan_working_tree() -> list[str]:
    findings = []
    for rel in git("ls-files").splitlines():
        if not rel or rel.endswith(ALLOWED_SUFFIXES):
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        for pattern, label in PATTERNS:
            if pattern.search(data):
                findings.append(f"working tree: {rel}: {label}")
    return findings


def scan_history() -> list[str]:
    listing = git(
        "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectname)"
    )
    blobs = [
        line.split()[1] for line in listing.splitlines() if line.startswith("blob")
    ]
    findings = []
    for oid in blobs:
        data = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "blob", oid],
            capture_output=True,
        ).stdout
        for pattern, label in PATTERNS:
            if pattern.search(data):
                findings.append(f"history blob {oid[:12]}: {label}")
    return findings, len(blobs)


def main() -> int:
    tracked = git("ls-files").splitlines()
    findings = []

    if ".env" in tracked:
        findings.append("`.env` is tracked by git")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if ".env" not in gitignore:
        findings.append("`.env` is missing from .gitignore")

    findings += scan_working_tree()
    history_findings, blob_count = scan_history()
    findings += history_findings

    print(f"scanned {len(tracked)} tracked files and {blob_count} history blobs")
    if findings:
        print(f"\nFAILED: {len(findings)} secret finding(s)")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("PASS: no secrets in the working tree or anywhere in history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
