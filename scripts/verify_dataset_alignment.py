"""Verify the published per-item evidence really indexes the pinned ChartQA test split.

This repository does not redistribute ChartQA queries, gold labels or chart
images. What it publishes instead is per-item correctness plus a content-free
SHA-256 identifier for each query. This script closes the loop: download the pinned dataset
revision yourself, hash its queries, and confirm the published evidence lines up
with the real test set. The hashes are alignment identifiers, not anonymization.

Opt-in on purpose — it needs network access and the `datasets` extra, so it is
not part of CI:

    uv sync --python 3.12 --extra data
    uv run --extra data python scripts/verify_dataset_alignment.py

Exit code 0 means every published query hash was found in the pinned split with
the expected multiplicity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"

EVIDENCE = (
    "assets/eval/per_item_baseline.json",
    "assets/eval/per_item_finetuned.json",
)

# ClassLabel in HuggingFaceM4/ChartQA: 0 = human, 1 = machine ("augmented")
SPLIT_CODE = {"human": 0, "augmented": 1}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--split", default="test", help="ChartQA split to check against (default: test)"
    )
    args = parser.parse_args()

    from chartqa_data import load_chartqa  # noqa: E402  (import after sys.path setup)

    print(f"loading HuggingFaceM4/ChartQA split={args.split} revision={args.revision}")
    failures = 0

    dataset_hashes: dict[str, Counter] = {}
    for name, code in SPLIT_CODE.items():
        rows = load_chartqa(args.split, human_or_machine=code, revision=args.revision)
        dataset_hashes[name] = Counter(sha256_text(q) for q in rows["query"])
        print(f"  {name}: {len(rows)} rows from the pinned dataset")

    for rel in EVIDENCE:
        doc = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        print(f"\n{rel}")
        for name in SPLIT_CODE:
            block = doc["splits"][name]
            published = Counter(item["query_sha256"] for item in block["items"])
            available = dataset_hashes[name]

            if block["n"] != sum(available.values()):
                print(
                    f"  FAIL {name}: evidence has n={block['n']}, "
                    f"pinned dataset has {sum(available.values())}"
                )
                failures += 1
                continue

            missing = published - available
            extra = available - published
            if missing:
                sample = list(missing)[:3]
                print(
                    f"  FAIL {name}: {sum(missing.values())} hash(es) not in the pinned split, e.g. {sample}"
                )
                failures += 1
            elif extra:
                print(
                    f"  FAIL {name}: {sum(extra.values())} dataset row(s) unaccounted for"
                )
                failures += 1
            else:
                print(
                    f"  ok   {name}: all {block['n']} query hashes match the pinned split"
                )

    print()
    if failures:
        print(f"FAILED: {failures} alignment check(s) did not match")
        return 1
    print("PASS: published evidence is aligned with the pinned ChartQA test split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
