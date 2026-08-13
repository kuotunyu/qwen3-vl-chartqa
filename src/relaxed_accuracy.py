"""Relaxed accuracy for ChartQA.

Canonical definition from the ChartQA paper (Masry et al., 2022), identical to
the implementation used by lmms-eval and the original vis-nlp/ChartQA repo:

- numeric answers: correct if |pred - target| / |target| <= 5%
- non-numeric answers: case-insensitive exact match
- a "%" suffix is stripped before float conversion (both sides divided by 100,
  which cancels out in the relative change)

Kept byte-for-byte compatible with the canonical logic so our scores are
comparable with published numbers; model-output cleanup lives in
`normalize_prediction`, separate from the metric itself.
"""

from __future__ import annotations


def _to_float(text: str) -> float | None:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def relaxed_correctness(
    prediction: str, target: str, max_relative_change: float = 0.05
) -> bool:
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    # NOTE: `target_float` (not `is not None`) is the canonical check — a
    # target of exactly 0 falls through to string match, as in the paper code.
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    return prediction.lower() == target.lower()


def normalize_prediction(text: str) -> str:
    """Light cleanup of model output before scoring.

    Only trims whitespace and a single trailing period — anything more
    aggressive would make scores incomparable with published results.
    """
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def relaxed_accuracy(predictions: list[str], targets: list[str]) -> float:
    assert len(predictions) == len(targets)
    if not predictions:
        return 0.0
    correct = sum(
        relaxed_correctness(normalize_prediction(p), t)
        for p, t in zip(predictions, targets, strict=True)
    )
    return correct / len(predictions)
