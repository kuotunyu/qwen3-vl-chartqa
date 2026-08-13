# Design Notes

## Goal and scope

This project demonstrates an auditable VLM optimization path rather than a single training score. The intended outcome is a chart-question-answering model with traceable training, a predefined quantization quality gate, and a serving benchmark whose workload controls are explicit enough to reproduce.

Non-goals:

- claiming state-of-the-art ChartQA performance;
- treating a 64-request benchmark as a production SLA;
- running 8B-model inference on the local RTX 2050 4GB machine;
- hiding latency regressions behind aggregate throughput numbers.

## Key decisions

### Qwen3-VL-8B-Instruct

Qwen3-VL-8B-Instruct was selected because the Unsloth, Transformers, llm-compressor, and vLLM paths were mature enough to complete the full workflow. The project values a reproducible end-to-end result over changing to a newer but less verified checkpoint midstream.

### ChartQA and paired evaluation

ChartQA provides a 2,500-question test split with equal human and augmented subsets. Every reported quality delta is paired within one evaluation recipe. Absolute scores from the Transformers/LoRA evaluation and the isolated-vLLM merged/AWQ evaluation are not cross-compared because their inference stacks differ.

### Relaxed accuracy: canonical zero-target behaviour

`relaxed_correctness` tests the truthiness of the parsed target, not `is not None`:

```python
if prediction_float is not None and target_float:
```

A target of exactly `0` is therefore falsy and falls through to case-insensitive
string comparison instead of the 5% relative-error branch. This is deliberate
compatibility with the reference implementation (Masry et al., 2022; the same
logic used by Pix2Struct and lmms-eval), and it also avoids a division by zero.
Changing it would make these scores incomparable with published numbers.

The behaviour was audited against the raw evidence before that evidence was
withdrawn from publication: 8 zero-target items exist across the six evaluation
sets, and canonical string matching and an explicit `prediction == 0` rule agree
on all 8. No headline metric depends on the choice. The semantics are pinned by
`tests/test_relaxed_accuracy.py` so that a future "cleanup" cannot silently move
the numbers.

### QLoRA recipe

The formal run used 15,000 shuffled examples for one epoch, rank-16 LoRA, effective batch size 16, and a 2,048-token maximum length. Vision and language paths were both adapted because chart understanding depends on visual encoding as well as answer generation.

The overall gain was +0.56 pp. Most improvement came from the augmented subset; the human subset improved only +0.16 pp. This is retained as a limitation rather than averaged away.

### AWQ W4A16 g32

Only language `Linear` weights were quantized to symmetric 4-bit groups of 32. The vision tower and `lm_head` remained at their original precision. Calibration used 256 ChartQA training examples with a 2,048-token cap.

The deployment gate was defined before the formal full-test run: AWQ could lose at most 2 percentage points versus merged 16-bit. The observed loss was 0.72 pp; paired bootstrap 95% CI was `[-1.40, -0.04] pp`, so the gate passed without re-quantization.

## Serving benchmark design

### Why fixed 64-token output

Natural ChartQA answers are often one token. TPOT estimated from such short generations is unstable and mostly reflects startup noise. Both warmup and measured requests therefore force 64 output tokens with `temperature=0` and `ignore_eos=true`.

This creates a controlled decode benchmark, not a simulation of the natural answer-length distribution. Because output length is fixed, requests/s and output tokens/s have the same relative ordering and are not independent pieces of evidence.

### Same cohort at every concurrency

All concurrency levels use the same 32 human + 32 augmented source questions in the same order. Prompt text and image dimensions are unchanged. This prevents chart size, prompt length, or question difficulty from being mistaken for a concurrency effect.

### Multimodal cache isolation

ChartQA may contain multiple questions for the same underlying image. The source cohort is therefore deduplicated by standardized-image SHA. Each concurrency level receives a tiny 12×12 corner-pixel variant, and warmup uses a separate variant. The notebook verifies uniqueness for all 512 encoded-file hashes and decoded-pixel hashes.

The vLLM multimodal processor cache and prefix cache are disabled. This ensures the benchmark measures vision processing rather than accidental reuse of precomputed image embeddings.

### Warmup and JIT handling

Every measured level is preceded by 64 independent warmup images with the same 64-token decode. vLLM's verbose JIT monitor remains enabled. JIT warnings are allowed during warmup but rejected inside the measured log window; a rejected level is not checkpointed or published.

The archived server logs consequently contain expected warmup JIT messages. `validity.jit_warning_levels=[]` means none appeared inside a formal measured window; it does not mean the entire server lifetime compiled no kernels.

### Process and identity isolation

Merged 16-bit and AWQ run in independent server process groups. Before and after each server, the notebook verifies port availability, `/health`, GPU compute processes, and process-group cleanup.

Checkpoint identity includes:

- driver revision and recipe SHA;
- model repo, pinned revision, and model fingerprint;
- pinned dataset revision;
- literal JSONL and workload hashes;
- quality-gate source SHA;
- GPU model, physical GPU UUID, and NVIDIA driver;
- vLLM version, concurrency, prompt count, and output length.

This permits same-runtime continuation but prevents results from different physical Colab GPUs from being combined into one canonical report.

### Publication gate

The final result is published only if all eight levels succeed, output lengths are exactly 64, input token workloads match across both models and all levels, cache controls are active, no measured-window JIT is present, and all checkpoints belong to the same GPU.

Archive and canonical result/table/plot/manifest/gate-source files are uploaded in one Hugging Face commit so a network interruption cannot leave a mixed old/new report.

## Reproducibility identifiers

- Benchmark Run ID: `v2-aa4442870cfd`
- Recipe SHA-256: `aa4442870cfd344275bbcd5f624ea74632aee23343e1d9bb518af56f23fa4eff`
- Workload SHA-256: `5c28e1ea96f35067b925f682ab21d07691d86b7eb918ac89ea5942f007b1c7dc`
- Quality-gate SHA-256: `decd099d7c83065a53e36450c04d06fd96446e53e40dec9ba99e5a7cee4b1f22`
- ChartQA revision: `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`
- Merged revision: `519060ef43df3261e0512e5ae4c82a4d4e675f32`
- Original benchmark revision (retired private archive): `e81d9332446307adc1b219ed326c8e55cead9015`
- Public-equivalent AWQ weight revision: `43b71926a1d645133560347787539729bcd3de6b`
- GPU: NVIDIA A100-SXM4-40GB; driver `580.82.07`
- vLLM: `0.25.1+cu129`; torch: `2.11.0+cu129`

## Operational lessons

1. An unpinned `vllm` install selected a CUDA 13 wheel and failed with missing `libcudart.so.13`. The formal notebooks now pin the verified CUDA 12.9 wheel stack before any torch import.
2. Printing full model logs into Colab made the browser appear frozen. Long operations now write detailed logs to disk and emit one heartbeat per minute.
3. A 100-question-per-split quantization check showed a misleading -3 pp drop. The complete 2,500-question paired run measured -0.72 pp, demonstrating why a small sample should not trigger an expensive re-quantization decision.
4. Reusing different questions at each concurrency confounds serving curves. The final benchmark fixes the source cohort and token workload.
5. A generic server health check is insufficient. A real multimodal chat probe is required before timing.
6. Saving checkpoints before inspecting the measured log can preserve invalid JIT-contaminated results. The final driver checks the measured log first.
7. Starting `vllm.LLM` directly inside a Colab kernel let the engine child inherit ipykernel's `stdout`, whose missing `fileno()` broke distributed initialization. The demo now starts the already benchmarked `vllm serve` path as an isolated OS process, logs to disk, performs a real multimodal probe, and lets Gradio call its localhost OpenAI endpoint.
8. Qualitative examples derived from ChartQA were removed from every public surface. Final live verification fetched a pinned row at runtime and compared inference with the label in memory; neither the chart, question, label, nor model output is published.
9. Mutable quality-gate files need a trusted content hash. Merely recording the current hash would accept an altered file under a new Run ID.
10. Canonical files should be committed atomically, not uploaded one by one.

## Deployment interpretation

### GGUF export and persistent Space

The fine-tuned merged model was exported with pinned llama.cpp commit `79bba02a6741de194912d370015866414faa83ad`. The published GGUF revision is `5e5860f5d4060f87614ecfb6243a001067979276`, containing a 5,027,784,800-byte `Q4_K_M` text model and a 752,289,728-byte `Q8_0` multimodal projector. An independent CPU smoke test fetched one pinned ChartQA row at runtime and passed an exact-match check without publishing the sample or answer.

The persistent Docker Space source remains ready, but the authenticated Hub API returned `402 Payment Required` while creating a Gradio/Docker `cpu-basic` Space and requested a Hugging Face PRO subscription. No paid resource was created. The selected no-cost architecture is now live as `steven0226/qwen3vl-chartqa-demo@29a6f0e096a5e93d7a9eed5083efaceca0c5af4c`: a persistent Static Space presents the evidence and directly opens the hosted demo notebook in Colab. The Static Space does not pretend to run the 8B model in-browser; GPU inference remains on Colab A100.

- AWQ is the default artifact because it passes the quality gate, cuts weight files by 56.9%, raises throughput at every tested concurrency, and lowers TPOT/E2E p95 at every level.
- Concurrency 4 is a responsive multi-user setting.
- Concurrency 8 is the practical balance for this controlled workload.
- Concurrency 16 maximizes aggregate throughput but raises AWQ TTFT p95 to about 958 ms and E2E p95 to about 1.61 s.
- The benchmark does not demonstrate lower vLLM-reserved VRAM: both servers reserve about 36 GB because both use `gpu_memory_utilization=0.88`.

### Archived CPU inference service (built, tested, never deployed)

`space/` holds a complete Docker inference service: FastAPI plus Gradio in front
of a supervised `llama.cpp` server running the pinned GGUF pair, with a single
active inference slot, a bounded waiting queue, per-IP sliding-window rate
limiting, input ceilings, `/healthz` and `/readyz` split, `/metrics` in
Prometheus text format, structured JSON logging that never records questions or
image payloads, and a bounded automatic restart budget.

| Control | Default |
|---|---:|
| Active inference | 1 |
| Waiting queue | 6 |
| Queue wait timeout | 180 s |
| Inference timeout | 360 s |
| Per-IP rate limit | 12 requests / 600 s |
| Question length | 800 characters |
| Source image | 16 MP |
| Encoded image payload | 8 MB |
| Output | 256 tokens |
| Restart budget | 3 / hour |

**It was never deployed.** Creating a Gradio or Docker Space on `cpu-basic`
requires a Hugging Face PRO subscription, and no paid resource was created. The
admission-control behaviour is covered by `tests/test_space_runtime.py`, which
verifies queue saturation, per-client rate-limit isolation, input rejection
before any call reaches `llama.cpp`, and gauge cleanup after completion. The
companion operational scripts (`scripts/evaluate_ood_space.py`,
`scripts/soak_test_space.py`, `scripts/monitor_space.py`) pass dry-run but have
never executed against a live deployment, so no OOD or soak result is claimed
anywhere in this repository.

The delivered no-cost artifact is the persistent Static Space plus the one-click
Colab A100 notebook.

## Publication boundary

This repository does not redistribute ChartQA. Published evidence is per-item
correctness with a content-free `query_sha256` identifier, never the query text, gold labels,
chart images, or raw prediction strings. `scripts/verify_claims.py` recomputes
every headline number from that evidence and also enforces the boundary, so a
regression fails CI rather than shipping. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Interview Q&A

### 1. Why use relaxed accuracy?

Chart answers often contain numbers where formatting or small rounding differences should not count as conceptual errors. The metric permits 5% relative error for numeric answers and uses normalized exact matching for non-numeric answers.

### 2. Why report human and augmented subsets separately?

Their difficulty and distribution differ substantially. A single overall score would hide that the fine-tuning gain came primarily from augmented questions.

### 3. Why evaluate AWQ against merged 16-bit instead of the original base?

The quantization question is whether compression preserves the already fine-tuned model. Merged 16-bit is therefore the correct paired reference.

### 4. Why group size 32?

It provides finer per-group scales than larger groups and was a conservative quality-oriented choice for a multimodal 8B model, at some metadata/compute cost.

### 5. Why leave the vision tower unquantized?

The vision stack is a small share of total weights, while quantizing it introduces disproportionate visual-quality risk. Most storage savings come from language linear layers.

### 6. Why does AWQ improve TPOT but sometimes worsen TTFT?

Lower-bit weight movement accelerates repeated decode steps. At higher concurrency, scheduling, multimodal preprocessing, batching, and quantized-kernel setup can add prefill/queueing overhead before the first token.

### 7. Why not trust the n=100 quality check?

Its observed -3 pp was driven by only a few discordant questions. The complete paired test reduced sampling uncertainty and changed the deployment decision.

### 8. Why force image hashes to be unique?

Without unique encoded content, vLLM may reuse multimodal processor results. That would turn the benchmark into a cache-hit test rather than a model serving test.

### 9. Why is p95 marked exploratory?

With 64 observations, p95 is determined by only a few tail samples. It is useful for direction, but production claims need repeated runs and a much larger request population.

### 10. What would be added before production?

Repeated randomized trials, realistic output-length distributions, longer soak tests, request admission/backpressure, telemetry, failure injection, and evaluation on out-of-domain charts and OCR-heavy inputs.
