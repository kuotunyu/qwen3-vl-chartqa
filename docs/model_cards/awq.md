---
license: apache-2.0
base_model: steven0226/qwen3vl-8b-chartqa-merged-16bit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3-vl
- vision-language
- chartqa
- awq
- compressed-tensors
- vllm
- 4-bit
datasets:
- HuggingFaceM4/ChartQA
language:
- en
---

# Qwen3-VL-8B ChartQA — AWQ W4A16 g32

Deployment-oriented 4-bit checkpoint of a ChartQA-fine-tuned Qwen3-VL-8B-Instruct model. The language `Linear` weights use symmetric AWQ W4A16 with group size 32; the vision tower and `lm_head` retain their original precision.

## Lineage

1. Base: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
2. QLoRA adapter: [`steven0226/qwen3vl-8b-chartqa-lora`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-lora)
3. Merged reference: [`steven0226/qwen3vl-8b-chartqa-merged-16bit`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-merged-16bit)
4. This AWQ checkpoint

Formal quantization used an NVIDIA A100-SXM4-40GB, 256 ChartQA calibration examples, maximum calibration length 2,048, seed 3407, llm-compressor 0.12.0, and `SMOKE_TEST=false`.

Weight files are 7.55 GB versus 17.53 GB for merged 16-bit: 56.9% smaller, or 2.32× compression.

## Quality gate

Complete ChartQA test set, relaxed accuracy with 5% numeric tolerance, isolated vLLM subprocesses:

| Split | n | Merged 16-bit | AWQ W4A16 g32 | Change |
|---|---:|---:|---:|---:|
| Human | 1,250 | 77.28% | 76.56% | -0.72 pp |
| Augmented | 1,250 | 95.20% | 94.48% | -0.72 pp |
| Overall | 2,500 | 86.24% | **85.52%** | **-0.72 pp — PASS** |

The predefined maximum degradation was 2 pp. Paired bootstrap 95% CI for the overall change was `[-1.40, -0.04] pp`.

Public-equivalent weight revision: `43b71926a1d645133560347787539729bcd3de6b`.

## A100 vLLM benchmark

Run `v2-aa4442870cfd`; one A100-SXM4-40GB; vLLM `0.25.1+cu129`; same 64 ChartQA requests per level; 64 warmup + 64 measured requests; fixed 64-token output; all 8 levels completed with zero failures.

| Concurrency | Merged tok/s | AWQ tok/s | AWQ throughput | Merged E2E p95 | AWQ E2E p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 67.29 | **123.24** | +83.2% | 1,007.16 ms | **562.00 ms** |
| 4 | 231.02 | **356.95** | +54.5% | 1,169.61 ms | **776.13 ms** |
| 8 | 387.58 | **528.48** | +36.4% | 1,426.90 ms | **1,038.69 ms** |
| 16 | 595.06 | **701.90** | +18.0% | 2,005.45 ms | **1,612.51 ms** |

![Latency and throughput benchmark](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq/resolve/main/bench/latency_throughput.png)

AWQ lowered TPOT p95 and E2E p95 at every tested concurrency. The tradeoff is TTFT p95 at concurrency 4/8/16, which increased by 9.0%/20.9%/18.8%.

Full artifacts: [`bench/benchmark_results.json`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq/blob/main/bench/benchmark_results.json), [`bench/benchmark_table.md`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq/blob/main/bench/benchmark_table.md), and [`eval_quant/results.json`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq/blob/main/eval_quant/results.json).

## vLLM usage

Use Linux with the verified CUDA 12.9 stack. Install vLLM before importing torch:

```bash
pip install -U uv
uv pip install --system --no-cache \
  "vllm==0.25.1+cu129" "torch==2.11.0+cu129" \
  "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.25.1/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

```python
import base64
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "steven0226/qwen3vl-8b-chartqa-awq"
REVISION = "43b71926a1d645133560347787539729bcd3de6b"

image_path = Path("chart.png")
image_uri = (
    "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode()
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {
                "type": "text",
                "text": "Which category has the highest value?\nAnswer the question using a single word or phrase.",
            },
        ],
    }
]

llm = LLM(
    model=MODEL,
    revision=REVISION,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1, "video": 0},
)
params = SamplingParams(temperature=0, max_tokens=64)
answer = llm.chat([messages], sampling_params=params)[0].outputs[0].text
print(answer)
```

## Intended use

- Chart question answering and controlled chart-reading experiments
- GPU serving with vLLM on compatible NVIDIA hardware
- Research and portfolio demonstrations of QLoRA → AWQ deployment

## Limitations

- Primarily evaluated on English ChartQA short answers; not validated for general documents or safety-critical use.
- Fine-tuning gains were concentrated on the augmented subset.
- Benchmark outputs were forced to 64 tokens; each level had only 64 measured requests, so p95 is exploratory.
- vLLM's ~36 GB reserved memory in this run was controlled by `gpu_memory_utilization=0.88` and is not evidence of reduced AWQ VRAM reservation.
- Production deployment should add repeated trials, realistic output lengths, and soak testing.

## 中文摘要

這是 ChartQA 微調後 Qwen3-VL-8B 的 AWQ W4A16 g32 部署版本。語言模型 Linear 權重採 4-bit symmetric group quantization；vision tower 與 `lm_head` 維持原精度。完整 2,500 題評估為 85.52%，相較 merged 16-bit 下降 0.72 個百分點，通過預設 2 pp 品質門檻。權重檔由 17.53 GB 降至 7.55 GB；A100 vLLM benchmark 在所有測試 concurrency 都提升吞吐並降低 TPOT/E2E p95，但高併發 TTFT p95 較高。
