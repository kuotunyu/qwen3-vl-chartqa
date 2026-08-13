# Third-party notices and publication boundary

This file records what this repository contains, what it deliberately does not
contain, and which third-party terms apply to each part. It is a description of
a conservative publication boundary, **not legal advice**.

## 1. This repository's own code — MIT

All source code, notebooks, scripts, documentation and synthetic assets authored
for this project are MIT licensed. See [LICENSE](LICENSE).

That includes the synthetic chart suite in `assets/ood_charts/`, which is drawn
deterministically by [`scripts/build_ood_chart_suite.py`](scripts/build_ood_chart_suite.py)
from hard-coded values. No third-party imagery is involved.

## 2. Qwen3-VL — Apache License 2.0

The base model is `Qwen/Qwen3-VL-8B-Instruct` (Qwen team / Alibaba Cloud),
released under the Apache License 2.0. Training used the pre-quantized
`unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` checkpoint.

The published derivatives — LoRA adapter, merged 16-bit, AWQ W4A16 and GGUF —
are hosted on the Hugging Face Hub and carry their own model cards. They are
weight artifacts and are not vendored into this repository.

## 3. ChartQA — not redistributed by this repository

Training and evaluation used the `HuggingFaceM4/ChartQA` dataset.

Two things make its redistribution terms unclear enough to avoid:

- the Hugging Face dataset card declares the tag `gpl-3.0`, but the card body is
  a placeholder (`More Information needed`) and carries no licence text;
- the underlying chart images were sourced from third-party publishers such as
  Statista, Pew Research Center, OECD and Our World in Data, each with its own
  copyright and terms.

GPL-3.0 is a software licence, and applying it to a dataset whose images
originate from those publishers leaves the downstream position ambiguous. This
repository therefore takes the conservative route and **redistributes no part of
ChartQA**: no chart images, no question text, no gold labels, and no raw model
predictions (a prediction paired with a correctness flag would reconstruct the
gold label for every correct item).

### What is published instead

| Evidence | Contents | Location |
|---|---|---|
| Aggregate results | split, n, relaxed accuracy | `assets/eval/results.json`, `assets/eval_quant/results.json` |
| Per-item correctness | positional index, content-free `query_sha256` identifier, correctness flag | `assets/eval/per_item_*.json`, `assets/eval_quant/per_item_*.json` |
| Benchmark workload | sample id, split, source index, `query_sha256`, image hashes | `assets/bench/workload_manifest.json` |

`query_sha256` is a SHA-256 of the UTF-8 query string. It avoids reproducing the
query text directly and lets a holder of the pinned dataset confirm alignment.
It is **not anonymization or a privacy guarantee**: because ChartQA is a known
public corpus, a holder can map hashes back to its query strings by comparison.

### Verifying the evidence yourself

- Offline, no dataset needed — recompute every published number and re-check the
  evidence hash chain:

  ```bash
  python scripts/verify_claims.py
  ```

- With your own copy of the pinned dataset — confirm the published per-item
  evidence indexes the real ChartQA test split:

  ```bash
  uv run --extra data python scripts/verify_dataset_alignment.py
  ```

The pinned dataset revision is `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`.

### Running the notebooks

The Colab notebooks download ChartQA under the dataset's own terms and write
derived artifacts (including chart images and predictions) into your own
runtime and, if you enable the upload cells, into your own Hugging Face
repositories. Deciding whether those derived artifacts may be published is the
responsibility of whoever runs them; this repository only declines to publish
its own copies.

One notebook pins the identity of a single ChartQA row for a smoke test. It does
so by SHA-256 (`notebooks/_build_gguf_notebook.py`) rather than by embedding the
question, and fetches the text at runtime from the pinned revision.

## 4. Other dependencies

Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`,
each under its own licence. `space/Dockerfile` builds `llama.cpp`
(pinned commit `79bba02a6741de194912d370015866414faa83ad`, MIT) from source.

## 5. Reported numbers

Every accuracy, latency and throughput figure in the READMEs is derived from the
machine-readable evidence in `assets/` and is re-verified on every CI run by
`scripts/verify_claims.py`. Absolute scores from the two evaluation stacks are
paired within each table and are not comparable across tables.
