"""ChartQA loading and Qwen3-VL chat-message formatting.

Shared between the local smoke tests and the Colab notebooks (training and
evaluation import from here so the prompt format is identical everywhere).

Dataset: HuggingFaceM4/ChartQA
  columns: image (PIL), query (str), label (list[str], len 1),
           human_or_machine (ClassLabel: 0=human, 1=machine/augmented)
"""

from __future__ import annotations

from typing import Any

DATASET_ID = "HuggingFaceM4/ChartQA"

# ChartQA convention (used by Qwen / lmms-eval): force short answers so that
# predictions are comparable under relaxed accuracy. The SAME instruction must
# be used at train and eval time.
ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."

HUMAN = 0
MACHINE = 1  # a.k.a. "augmented" in the ChartQA paper


def _download_with_retry(
    filename: str,
    revision: str | None = None,
    max_retries: int = 6,
    base_delay: float = 5.0,
) -> str:
    """hf_hub_download with retries.

    HF's Xet CDN backend has intermittently returned "invalid key pair
    id" 403s from Colab/GCP — observed even for a file that downloaded
    fine moments earlier in the same session, so this isn't a broken
    file, it's a flaky signed URL. A fresh request gets a fresh signed
    URL and retrying has recovered every time it's been tried. Retries
    print progress so a wait here reads as "working on it", not a
    repeat of the earlier silent-hang bugs.
    """
    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(1, max_retries + 1):
        try:
            return hf_hub_download(
                DATASET_ID, filename, repo_type="dataset", revision=revision
            )
        except HfHubHTTPError as e:
            if attempt == max_retries:
                raise
            wait = base_delay * attempt
            print(
                f"  [{filename}] attempt {attempt}/{max_retries} failed "
                f"({type(e).__name__}); retrying in {wait:.0f}s"
            )
            time.sleep(wait)


def load_chartqa(
    split: str,
    n: int | None = None,
    human_or_machine: int | None = None,
    seed: int = 42,
    revision: str | None = None,
):
    """Load a ChartQA split, optionally filtered and subsampled.

    Downloads only the requested split's parquet file(s) directly via
    hf_hub_download (a plain sequential full-file download, retried on
    failure — see _download_with_retry) instead of datasets.load_dataset().
    Two other paths were tried and both hit Hugging Face's Xet CDN
    backend returning invalid-signature 403s on Colab: the eager/
    non-streaming path (which prepares every split up front, so a
    broken file in an unrelated split blocks callers that never needed
    it) and the streaming path (whose Parquet reader does byte-range
    reads through HfFileSystem, a different request shape that hit the
    same CDN bug on a *different* file). The signing failures turned
    out to be intermittent rather than tied to one broken file, so the
    download itself now retries with a fresh signed URL each time.

    human_or_machine: HUMAN (0), MACHINE (1), or None for both.
    n: if set, shuffle with `seed` and keep the first n rows.
    revision: pin a dataset revision (commit SHA); None tracks the default branch.
    """
    from datasets import Dataset
    from huggingface_hub import HfApi

    files = sorted(
        f
        for f in HfApi().list_repo_files(
            DATASET_ID, repo_type="dataset", revision=revision
        )
        if f.startswith(f"data/{split}-")
    )
    if not files:
        raise ValueError(f"no data files found for split={split!r}")
    local_paths = [_download_with_retry(f, revision=revision) for f in files]

    ds = Dataset.from_parquet(local_paths)
    if human_or_machine is not None:
        ds = ds.filter(lambda ex: ex["human_or_machine"] == human_or_machine)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def get_answer(example: dict[str, Any]) -> str:
    """ChartQA labels are single-element lists."""
    label = example["label"]
    if isinstance(label, list):
        assert len(label) == 1, f"unexpected multi-answer label: {label!r}"
        return str(label[0])
    return str(label)


def build_user_content(query: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": f"{query}\n{ANSWER_INSTRUCTION}"},
    ]


def to_messages(example: dict[str, Any], include_answer: bool = True) -> list[dict]:
    """One ChartQA row -> Qwen3-VL chat messages.

    The PIL image is embedded in the user content (the format unsloth's
    UnslothVisionDataCollator and transformers' AutoProcessor both accept).
    include_answer=False yields an inference prompt (no assistant turn).
    """
    user_content = [{"type": "image", "image": example["image"]}]
    user_content += build_user_content(example["query"])
    messages = [{"role": "user", "content": user_content}]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": get_answer(example)}],
            }
        )
    return messages


def convert_to_conversation(example: dict[str, Any]) -> dict[str, Any]:
    """Map-style wrapper for datasets: row -> {"messages": [...]}."""
    return {"messages": to_messages(example, include_answer=True)}
