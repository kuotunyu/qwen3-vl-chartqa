"""Verify every headline claim in the READMEs against machine-readable evidence.

CPU-only, offline, no model weights and no dataset download. Exits non-zero on
the first category that fails, so CI turns a stale number into a red build.

What it checks:

1. evidence integrity   - internal SHA-256 chain of the benchmark artifacts
2. per-item evidence    - recomputes every accuracy from per-item correctness
3. README claims        - parses the published tables and compares them
4. derived claims       - compression, throughput, TPOT, E2E and TTFT deltas
5. publication boundary - no ChartQA text, no secrets, no private artifacts

Run:  python scripts/verify_claims.py [-v]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

PCT_TOL = 0.005  # README prints percentages to 2 decimals
MS_TOL = 0.005
RATIO_TOL = 0.05  # README prints derived deltas to 1 decimal

SPLITS = ("human", "augmented")

# Files removed from this repository on purpose: ChartQA queries, gold labels
# and chart images are not redistributed. See THIRD_PARTY_NOTICES.md.
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"assets/eval/cases/"),
    re.compile(r"assets/eval/predictions_(baseline|finetuned)\.json$"),
    re.compile(r"assets/eval_quant/preds_.*\.json$"),
    re.compile(r"space_static/assets/case_\d+\.png$"),
)

# Each needle is assembled from fragments so this scanner does not flag itself,
# and so that a historical copy of it does not fail a later run. Keep the `+`
# concatenation: adjacent literals would be folded back together.
FORBIDDEN_CONTENT = (
    ('"quer' + 'ies"', "raw ChartQA query list"),
    ('"go' + 'lds"', "raw ChartQA gold-label list"),
    ("smallest value on the " + "blue bar", "verbatim ChartQA query text"),
    ("kuotu" + "nyh", "superseded private identity"),
)

SECRET_PATTERNS = (
    (re.compile(r"hf_[A-Za-z0-9]{34,}"), "Hugging Face token"),
    (re.compile(r"ghp_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{40,}"), "GitHub fine-grained token"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "OpenAI-style key"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{35}"), "Google API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
)


class Failures:
    def __init__(self, verbose: bool) -> None:
        self.items: list[str] = []
        self.checks = 0
        self.verbose = verbose

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        if ok:
            if self.verbose:
                print(f"    ok   {label}")
        else:
            self.items.append(f"{label}{(' — ' + detail) if detail else ''}")
            print(f"    FAIL {label}{(' — ' + detail) if detail else ''}")
        return ok

    def close(self, ok: bool, label: str, a, b, tol: float) -> bool:
        return self.check(
            abs(a - b) <= tol if ok else False, label, f"{a!r} vs {b!r} (tol {tol})"
        )

    def near(self, label: str, a: float, b: float, tol: float) -> bool:
        return self.check(abs(a - b) <= tol, label, f"{a} vs {b} (tol {tol})")


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha256_file(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True
    )
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line]


# --------------------------------------------------------------------------
# markdown table parsing
# --------------------------------------------------------------------------


def _num(cell: str) -> float | None:
    cleaned = (
        cell.replace("**", "")
        .replace(",", "")
        .replace("%", "")
        .replace("ms/tok", "")
        .replace("ms", "")
        .replace("pp", "")
        .replace("—", " ")
        .replace("PASS", "")
        .strip()
    )
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_tables(markdown: str) -> list[list[list[str]]]:
    """Return every markdown table as a list of cell-rows (header included)."""
    tables, current = [], []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue  # separator row
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def find_table(tables, *header_needles: str):
    for table in tables:
        header = " ".join(table[0]).lower()
        if all(needle.lower() in header for needle in header_needles):
            return table
    return None


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def check_evidence_integrity(f: Failures) -> None:
    print("[1/5] evidence integrity")
    bench = load_json("assets/bench/benchmark_results.json")
    manifest = load_json("assets/bench/workload_manifest.json")

    f.check(
        bench["quality_gate"]["source_sha256"]
        == sha256_file("assets/bench/quality_gate_source.json"),
        "quality_gate_source.json matches its recorded SHA-256",
    )
    f.check(
        bench["recipe"]["workload_sha256"] == manifest["workload_sha256"],
        "workload manifest SHA-256 matches the benchmark recipe",
    )
    f.check(
        bench["validity"]["valid_for_reporting"] is True,
        "benchmark is marked valid for reporting",
    )
    for key in (
        "all_requests_succeeded",
        "fixed_output_tokens_verified",
        "same_source_cohort_across_levels",
        "encoded_and_decoded_image_hashes_unique",
        "processor_cache_disabled",
        "prefix_cache_disabled",
        "same_physical_gpu",
    ):
        f.check(bench["validity"][key] is True, f"validity gate: {key}")
    f.check(
        bench["validity"]["jit_warning_levels"] == [],
        "no JIT warning inside any measured window",
    )

    f.check(len(bench["rows"]) == 8, "benchmark has 8 measured levels")
    for row in bench["rows"]:
        tag = f"{row['model']} c={row['concurrency']}"
        f.check(row["failed"] == 0, f"{tag}: zero failed requests")
        f.check(row["n"] == 64, f"{tag}: 64 measured requests")
        f.check(
            row["jit_warning_during_measurement"] is False,
            f"{tag}: no JIT during measurement",
        )

    gpus = {row["gpu_uuid"] for row in bench["rows"]}
    f.check(len(gpus) == 1, "all levels ran on one physical GPU", str(gpus))
    attempts = {row["attempt_id"] for row in bench["rows"]}
    f.check(len(attempts) == 1, "all levels belong to one attempt", str(attempts))

    sample_hashes = [
        s["standardized_image_sha256"]
        for level in manifest["levels"]
        for s in level["samples"]
        if "standardized_image_sha256" in s
    ]
    if sample_hashes:
        f.check(
            len(sample_hashes) == len(set(sample_hashes)),
            "every standardized image hash in the workload is unique",
        )


def _per_item(f: Failures, rel: str, expect_query_hash: bool) -> dict:
    doc = load_json(rel)
    name = Path(rel).name
    f.check(doc["evidence_kind"] == "per_item_correctness", f"{name}: evidence kind")
    for split in SPLITS:
        block = doc["splits"][split]
        items = block["items"]
        f.check(len(items) == block["n"], f"{name}/{split}: item count matches n")
        f.check(
            [i["idx"] for i in items] == list(range(block["n"])),
            f"{name}/{split}: indices are contiguous",
        )
        n_correct = sum(1 for i in items if i["correct"])
        f.check(
            n_correct == block["n_correct"], f"{name}/{split}: n_correct is consistent"
        )
        f.near(
            f"{name}/{split}: accuracy recomputed from per-item flags",
            n_correct / block["n"],
            block["relaxed_accuracy"],
            1e-9,
        )
        has_hash = all("query_sha256" in i for i in items)
        f.check(has_hash == expect_query_hash, f"{name}/{split}: query hash presence")
        if expect_query_hash:
            f.check(
                all(re.fullmatch(r"[0-9a-f]{64}", i["query_sha256"]) for i in items),
                f"{name}/{split}: query hashes are well formed",
            )
        leaked = {k for i in items for k in i} - {"idx", "correct", "query_sha256"}
        f.check(not leaked, f"{name}/{split}: no extra per-item fields", str(leaked))
    return doc


def check_per_item_evidence(f: Failures) -> dict[str, dict]:
    print("[2/5] per-item evidence recomputation")
    docs = {
        "baseline": _per_item(f, "assets/eval/per_item_baseline.json", True),
        "finetuned": _per_item(f, "assets/eval/per_item_finetuned.json", True),
        "merged16": _per_item(
            f, "assets/eval_quant/per_item_merged16_n1250.json", False
        ),
        "awq": _per_item(f, "assets/eval_quant/per_item_awq_n1250.json", False),
    }

    eval_results = load_json("assets/eval/results.json")
    quant_results = load_json("assets/eval_quant/results.json")

    for row in eval_results["table"]:
        split = row["test split"]
        for key, name in (
            ("baseline (before)", "baseline"),
            ("fine-tuned (after)", "finetuned"),
        ):
            got = (
                docs[name]["overall"]["relaxed_accuracy"]
                if split == "overall"
                else docs[name]["splits"][split]["relaxed_accuracy"]
            )
            f.near(f"eval results.json {split}/{name}", got, row[key], 1e-9)

    for row in quant_results["table"]:
        split = row["test split"]
        for key, name in (("merged-16bit", "merged16"), ("awq-w4a16-g32", "awq")):
            got = (
                docs[name]["overall"]["relaxed_accuracy"]
                if split == "overall"
                else docs[name]["splits"][split]["relaxed_accuracy"]
            )
            f.near(f"quant results.json {split}/{name}", got, row[key], 1e-9)

    f.check(
        quant_results["quality_gate_passed"] is True,
        "AWQ quality gate recorded as PASS",
    )
    drop_pp = (
        docs["awq"]["overall"]["relaxed_accuracy"]
        - docs["merged16"]["overall"]["relaxed_accuracy"]
    ) * 100
    f.check(
        abs(drop_pp) <= quant_results["quality_gate_max_drop_pp"],
        "recomputed AWQ drop is inside the predefined gate",
        f"{drop_pp:.4f} pp",
    )
    return docs


def check_readme_accuracy_tables(f: Failures, docs: dict[str, dict]) -> None:
    print("[3/5] README accuracy tables")
    for readme in ("README.md", "README.zh-TW.md"):
        tables = parse_tables((ROOT / readme).read_text(encoding="utf-8"))

        ft = find_table(tables, "n", "before") or find_table(tables, "n", "微調前")
        if not f.check(ft is not None, f"{readme}: fine-tuning table found"):
            continue
        for row in ft[1:]:
            split = row[0].lower().replace("整體", "overall").replace("人工", "human")
            key = (
                "overall"
                if "overall" in split
                else "human"
                if "human" in split
                else "augmented"
            )
            n = _num(row[1])
            before, after = _num(row[2]), _num(row[3])
            for name, want in (("baseline", before), ("finetuned", after)):
                got = (
                    docs[name]["overall"]
                    if key == "overall"
                    else docs[name]["splits"][key]
                )
                f.near(
                    f"{readme} fine-tuning {key}/{name}",
                    got["relaxed_accuracy"] * 100,
                    want,
                    PCT_TOL,
                )
                f.check(got["n"] == n, f"{readme} fine-tuning {key}: n={n}")

        qt = find_table(tables, "n", "awq")
        if not f.check(qt is not None, f"{readme}: quantization table found"):
            continue
        for row in qt[1:]:
            split = row[0].lower()
            key = (
                "overall"
                if ("overall" in split or "整體" in row[0])
                else "human"
                if "human" in split
                else "augmented"
            )
            for name, cell in (("merged16", row[2]), ("awq", row[3])):
                got = (
                    docs[name]["overall"]
                    if key == "overall"
                    else docs[name]["splits"][key]
                )
                f.near(
                    f"{readme} quantization {key}/{name}",
                    got["relaxed_accuracy"] * 100,
                    _num(cell),
                    PCT_TOL,
                )


def check_readme_benchmark(f: Failures) -> None:
    print("[4/5] README benchmark and derived claims")
    bench = load_json("assets/bench/benchmark_results.json")
    rows = {(r["model"], r["concurrency"]): r for r in bench["rows"]}

    label = {"Merged 16-bit": "merged-16bit", "AWQ W4A16 g32": "awq-w4a16-g32"}

    for readme in ("README.md", "README.zh-TW.md"):
        tables = parse_tables((ROOT / readme).read_text(encoding="utf-8"))
        bt = find_table(tables, "concurrency") or find_table(tables, "並發")
        if not f.check(bt is not None, f"{readme}: benchmark table found"):
            continue
        checked = 0
        for row in bt[1:]:
            model = label.get(row[0].replace("**", "").strip())
            if model is None:
                continue
            c = int(_num(row[1]))
            src = rows.get((model, c))
            if not f.check(src is not None, f"{readme}: row {model} c={c} exists"):
                continue
            f.near(
                f"{readme} {model} c={c} tok/s",
                src["output_tokens_per_s"],
                _num(row[2]),
                MS_TOL,
            )
            f.near(
                f"{readme} {model} c={c} TTFT p95",
                src["ttft_p95_ms"],
                _num(row[3]),
                MS_TOL,
            )
            f.near(
                f"{readme} {model} c={c} TPOT p95",
                src["tpot_p95_ms"],
                _num(row[4]),
                MS_TOL,
            )
            f.near(
                f"{readme} {model} c={c} E2E p95",
                src["e2e_p95_ms"],
                _num(row[5]),
                MS_TOL,
            )
            checked += 1
        f.check(
            checked == 8, f"{readme}: all 8 benchmark rows cross-checked", str(checked)
        )

    # derived claims
    merged_gb = bench["models"]["merged16"]["weight_bytes"] / 1000**3
    awq_gb = bench["models"]["awq"]["weight_bytes"] / 1000**3
    f.near("merged 16-bit weight size 17.53 GB", merged_gb, 17.53, 0.005)
    f.near("AWQ weight size 7.55 GB", awq_gb, 7.55, 0.005)
    f.near("weight reduction 56.9%", (1 - awq_gb / merged_gb) * 100, 56.9, 0.05)
    f.near("compression ratio 2.32x", merged_gb / awq_gb, 2.32, 0.005)

    expected = {
        1: (83.2, 51.9, 44.2, -3.7),
        4: (54.5, 39.4, 33.6, 9.0),
        8: (36.4, 32.4, 27.2, 20.9),
        16: (18.0, 17.4, 19.6, 18.8),
    }
    for comp in bench["comparisons"]:
        c = comp["concurrency"]
        thr, tpot, e2e, ttft = expected[c]
        f.near(
            f"c={c} throughput gain",
            (comp["awq_output_throughput_ratio"] - 1) * 100,
            thr,
            RATIO_TOL,
        )
        f.near(
            f"c={c} TPOT p95 reduction",
            (1 - comp["awq_tpot_p95_ratio"]) * 100,
            tpot,
            RATIO_TOL,
        )
        f.near(
            f"c={c} TTFT p95 change",
            (comp["awq_ttft_p95_ratio"] - 1) * 100,
            ttft,
            RATIO_TOL,
        )
        m = rows[("merged-16bit", c)]["e2e_p95_ms"]
        a = rows[("awq-w4a16-g32", c)]["e2e_p95_ms"]
        f.near(f"c={c} E2E p95 reduction", (1 - a / m) * 100, e2e, RATIO_TOL)


def check_publication_boundary(f: Failures) -> None:
    print("[5/5] publication boundary")
    files = tracked_files()
    f.check(bool(files), "git ls-files returned tracked files")

    for pattern in FORBIDDEN_PATH_PATTERNS:
        hits = [p for p in files if pattern.search(p)]
        f.check(not hits, f"no tracked file matches {pattern.pattern}", str(hits[:3]))

    f.check(".env" not in files, ".env is not tracked")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    f.check(".env" in gitignore, ".env is listed in .gitignore")

    text_ext = {
        ".py",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".html",
        ".ipynb",
        ".ps1",
        ".cfg",
        ".ini",
        ".lock",
        ".dockerignore",
    }
    content_hits: list[str] = []
    secret_hits: list[str] = []
    for rel in files:
        path = ROOT / rel
        if not path.exists() or path.suffix.lower() not in text_ext:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for needle, why in FORBIDDEN_CONTENT:
            if needle in body:
                content_hits.append(f"{rel}: {why}")
        for pattern, why in SECRET_PATTERNS:
            if pattern.search(body):
                secret_hits.append(f"{rel}: {why}")

    f.check(
        not content_hits,
        "no redistributed ChartQA text or stale identity",
        "; ".join(content_hits[:4]),
    )
    f.check(
        not secret_hits,
        "no secret-shaped literals in tracked files",
        "; ".join(secret_hits[:4]),
    )

    history = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--all", "--pretty=format:", "--name-only"],
        capture_output=True,
        text=True,
    )
    if history.returncode == 0:
        ever = {p for p in history.stdout.splitlines() if p}
        for pattern in FORBIDDEN_PATH_PATTERNS:
            hits = [p for p in ever if pattern.search(p)]
            f.check(not hits, f"history is free of {pattern.pattern}", str(hits[:3]))

    idents = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--all", "--format=%an <%ae>|%cn <%ce>"],
        capture_output=True,
        text=True,
    )
    if idents.returncode == 0:
        raw_idents = set(idents.stdout.split("\n")) - {""}
        human_authors = {
            ident.split("|")[0]
            for ident in raw_idents
            if "dependabot[bot]" not in ident and "github-actions[bot]" not in ident
        }
        f.check(
            len(human_authors) == 1,
            "git history carries exactly one identity",
            str(sorted(raw_idents)[:3]),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    f = Failures(args.verbose)
    check_evidence_integrity(f)
    docs = check_per_item_evidence(f)
    check_readme_accuracy_tables(f, docs)
    check_readme_benchmark(f)
    check_publication_boundary(f)

    print()
    if f.items:
        print(f"FAILED: {len(f.items)} of {f.checks} checks")
        for item in f.items:
            print(f"  - {item}")
        return 1
    print(f"PASS: {f.checks} claim checks verified against machine-readable evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
