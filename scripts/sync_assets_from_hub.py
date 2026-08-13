"""Pull training/eval/benchmark artifacts from HF Hub back into assets/.

The Colab notebooks push everything to the Hub (loss curve, eval results,
case images, benchmark tables); this script mirrors those files into the
local repo so Phase 6 README/docs can reference them from git.

Run locally:
    uv run python scripts/sync_assets_from_hub.py
    uv run python scripts/sync_assets_from_hub.py --user <hf-username>
    uv run python scripts/sync_assets_from_hub.py --with-bench   # after Phase 4

Needs a logged-in HF token (`hf auth login`) only when --user is omitted;
the repos themselves are public.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError

# .env's HF_TOKEN (if present) takes precedence over the cached CLI login;
# huggingface_hub picks it up from the environment automatically.
load_dotenv()

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# repo suffix -> (allow_patterns, dest subdir mapping)
ADAPTER_PATTERNS = ["loss_curve.png", "log_history.json", "eval/**"]
AWQ_PATTERNS = [
    "quantization_metadata.json",
    "recipe.yaml",
    "bench/**",
    "eval_quant/**",
]

# The Hub repos still carry the full ChartQA-derived evaluation output (chart
# images, queries, gold labels, raw predictions). This repository deliberately
# does not redistribute any of it — see THIRD_PARTY_NOTICES.md — so those files
# are dropped on the way in rather than being mirrored back into assets/.
# Without this guard, a routine sync would silently reopen the boundary.
EXCLUDED_FROM_SYNC = (
    "eval/cases/",
    "eval/predictions_",
    "eval_quant/preds_",
)


def is_excluded(rel: Path) -> bool:
    posix = rel.as_posix()
    return any(part in posix for part in EXCLUDED_FROM_SYNC)


def resolve_user(cli_user: str | None) -> str:
    if cli_user:
        return cli_user
    from huggingface_hub import whoami

    return whoami()["name"]


def copy_tree(src_root: Path) -> list[Path]:
    """Copy every file under src_root into assets/, preserving relative paths."""
    copied = []
    skipped = 0
    for f in sorted(src_root.rglob("*")):
        if not f.is_file() or f.name == ".gitattributes":
            continue
        rel = f.relative_to(src_root)
        if rel.parts[0] == ".cache":
            continue  # huggingface_hub's own local_dir bookkeeping, not a repo file
        if is_excluded(rel):
            skipped += 1
            continue

        dest = ASSETS / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied.append(dest)
    if skipped:
        print(f"   skipped {skipped} ChartQA-derived file(s) — not redistributed here")
    return copied


def sync_repo(repo_id: str, patterns: list[str]) -> list[Path]:
    print(f"== syncing {repo_id} ({', '.join(patterns)}) ==")
    try:
        # local_dir forces real file downloads instead of the default
        # cache/symlink layout — Windows accounts without Developer Mode
        # or admin rights can't create the symlinks that layout needs.
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(
                snapshot_download(repo_id, allow_patterns=patterns, local_dir=tmp)
            )
            copied = copy_tree(local)
    except RepositoryNotFoundError:
        print(
            f"   repo not found: https://huggingface.co/{repo_id}\n"
            "   -> has the corresponding Colab notebook been run (and pushed) yet?"
        )
        return []
    for p in copied:
        print(f"   -> {p.relative_to(ASSETS.parent)}")
    if not copied:
        print("   (nothing matched — has this stage been run on Colab yet?)")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user", help="HF username (default: whoami() via logged-in token)"
    )
    parser.add_argument(
        "--with-bench",
        action="store_true",
        help="also pull bench/ and eval_quant/ from the AWQ repo (Phase 4 outputs)",
    )
    args = parser.parse_args()

    user = resolve_user(args.user)
    ASSETS.mkdir(exist_ok=True)

    copied = sync_repo(f"{user}/qwen3vl-8b-chartqa-lora", ADAPTER_PATTERNS)
    if args.with_bench:
        copied += sync_repo(f"{user}/qwen3vl-8b-chartqa-awq", AWQ_PATTERNS)

    print(f"\nsynced {len(copied)} files -> {ASSETS}")


if __name__ == "__main__":
    main()
