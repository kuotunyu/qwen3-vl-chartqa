"""Phase 1 smoke test: ChartQA preprocessing + Qwen3-VL chat template.

Run locally (CPU only, no model weights):
    uv run python scripts/smoke_test_data.py

Checks:
1. dataset schema and the human_or_machine label mapping
2. row -> chat messages conversion (src/chartqa_data.py)
3. processor.apply_chat_template renders image + text correctly
4. relaxed accuracy metric edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.chartqa_data import (
    ANSWER_INSTRUCTION,
    DATASET_ID,
    HUMAN,
    MACHINE,
    convert_to_conversation,
    get_answer,
    load_chartqa,
    to_messages,
)
from src.relaxed_accuracy import (
    normalize_prediction,
    relaxed_accuracy,
    relaxed_correctness,
)

N = 24


def check_dataset() -> None:
    print(f"== 1. dataset schema ({DATASET_ID}) ==")
    ds = load_chartqa("train", n=N)
    names = ds.features["human_or_machine"].names
    assert names == ["human", "machine"], names
    print(f"   human_or_machine classes: {names} (0=human, 1=machine/augmented)")

    test = load_chartqa("test")
    n_human = sum(1 for v in test["human_or_machine"] if v == HUMAN)
    n_machine = sum(1 for v in test["human_or_machine"] if v == MACHINE)
    print(f"   test split: {len(test)} rows = {n_human} human + {n_machine} machine")
    assert n_human + n_machine == len(test)

    for ex in ds:
        assert isinstance(ex["image"], Image.Image)
        assert isinstance(ex["query"], str) and ex["query"]
        assert isinstance(get_answer(ex), str)
    print(f"   {N} sampled rows: image is PIL, query is str, label extracts OK")


def check_messages() -> dict:
    print("== 2. row -> chat messages ==")
    ds = load_chartqa("train", n=2)
    ex = ds[0]
    msgs = to_messages(ex)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    types = [c["type"] for c in msgs[0]["content"]]
    assert types == ["image", "text"], types
    assert isinstance(msgs[0]["content"][0]["image"], Image.Image)
    assert ANSWER_INSTRUCTION in msgs[0]["content"][1]["text"]
    assert msgs[1]["content"][0]["text"] == get_answer(ex)
    conv = convert_to_conversation(ex)
    assert "messages" in conv
    print(f"   query: {ex['query']!r}")
    print(f"   answer: {get_answer(ex)!r}")
    print("   structure OK (user: [image, text+instruction], assistant: [answer])")
    return ex


def check_chat_template(ex: dict) -> None:
    print("== 3. processor chat template ==")
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    msgs = to_messages(ex)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<|vision_start|>" in text and "<|vision_end|>" in text, text[:500]
    assert ex["query"] in text
    assert get_answer(ex) in text
    print("   rendered template:")
    for line in text.splitlines():
        print("   |", line)

    # Full processor call: template + image -> pixel tensors, as in training.
    inputs = proc(
        text=[proc.apply_chat_template(msgs, tokenize=False)],
        images=[ex["image"].convert("RGB")],
        return_tensors="pt",
    )
    print(f"   processor outputs: { {k: tuple(v.shape) for k, v in inputs.items()} }")
    assert "input_ids" in inputs and "pixel_values" in inputs


def check_metric() -> None:
    print("== 4. relaxed accuracy metric ==")
    cases = [
        ("38", "38", True),
        ("38.5", "38", True),  # 1.3% off -> ok
        ("40", "38", False),  # 5.3% off -> fail
        ("14%", "14", False),  # canonical: 0.14 vs 14 -> fail (as in paper code)
        ("14%", "14%", True),
        ("Yes", "yes", True),
        ("blue", "Blue", True),
        ("100", "0", False),  # zero target falls to string match
        ("0", "0", True),
    ]
    for pred, target, expected in cases:
        got = relaxed_correctness(pred, target)
        status = "OK " if got == expected else "FAIL"
        print(f"   [{status}] relaxed_correctness({pred!r}, {target!r}) = {got}")
        assert got == expected, (pred, target, got, expected)
    assert normalize_prediction(" 42. ") == "42"
    assert relaxed_accuracy(["38", "no"], ["38", "No"]) == 1.0
    print("   metric edge cases OK")


if __name__ == "__main__":
    check_dataset()
    ex = check_messages()
    check_chat_template(ex)
    check_metric()
    print("\nALL SMOKE TESTS PASSED")
