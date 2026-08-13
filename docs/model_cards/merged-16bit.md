---
license: apache-2.0
base_model: unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3-vl
- vision-language
- chartqa
- qlora
- merged
- vllm
datasets:
- HuggingFaceM4/ChartQA
language:
- en
---

# Qwen3-VL-8B ChartQA — Merged 16-bit

Full merged checkpoint produced by merging [`steven0226/qwen3vl-8b-chartqa-lora`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-lora) into its Qwen3-VL-8B base. This is the quality reference used before AWQ quantization and the 16-bit serving baseline.

## Training summary

- 15,000 shuffled ChartQA training examples
- 1 epoch; maximum sequence length 2,048
- LoRA rank 16, alpha 16, dropout 0
- Vision, language, attention, and MLP modules enabled for LoRA adaptation
- Effective A100 batch size 16
- 8-bit AdamW; peak learning rate `2e-4`; linear schedule
- Runtime 3,579 seconds; final training loss 0.5907

## Evaluation

Complete ChartQA test set, isolated vLLM evaluation, relaxed accuracy with 5% numeric tolerance:

| Split | n | Accuracy |
|---|---:|---:|
| Human | 1,250 | 77.28% |
| Augmented | 1,250 | 95.20% |
| Overall | 2,500 | **86.24%** |

Evaluation revision: `519060ef43df3261e0512e5ae4c82a4d4e675f32`.

The downstream [AWQ checkpoint](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq) scored 85.52%, a -0.72 pp change that passed the predefined -2 pp quality gate.

## Serving reference

The formal A100 benchmark used this checkpoint as the 16-bit baseline. Weight files total 17.53 GB.

| Concurrency | Output tok/s | TTFT p95 | TPOT p95 | E2E p95 |
|---:|---:|---:|---:|---:|
| 1 | 67.29 | 160.76 ms | 13.43 ms/tok | 1,007.16 ms |
| 4 | 231.02 | 299.78 ms | 15.04 ms/tok | 1,169.61 ms |
| 8 | 387.58 | 473.68 ms | 18.33 ms/tok | 1,426.90 ms |
| 16 | 595.06 | 806.77 ms | 24.14 ms/tok | 2,005.45 ms |

Complete comparison and reproducibility artifacts are in the [AWQ repo `bench/`](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq/tree/main/bench).

## Usage

This is a standard merged Qwen3-VL checkpoint. It can be loaded with a Qwen3-VL-compatible Transformers/vLLM environment. The verified serving stack was vLLM `0.25.1+cu129` with torch `2.11.0+cu129` on Linux A100.

```python
from vllm import LLM

llm = LLM(
    model="steven0226/qwen3vl-8b-chartqa-merged-16bit",
    revision="519060ef43df3261e0512e5ae4c82a4d4e675f32",
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1, "video": 0},
)
```

For a complete image/chat example, use the [AWQ model card](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq#vllm-usage) and replace the model ID/revision.

## Limitations

- Evaluated primarily on English ChartQA short answers.
- Not validated for general OCR documents, out-of-domain charts, or safety-critical use.
- This checkpoint is the larger quality reference; the AWQ checkpoint is recommended for serving when its -0.72 pp quality tradeoff is acceptable.

## 中文摘要

這是將 ChartQA LoRA adapter 合併回 Qwen3-VL-8B 後的 16-bit 完整模型，也是 AWQ 量化前的品質與 serving baseline。完整 2,500 題 vLLM 評估為 86.24%；權重檔 17.53 GB。若部署可接受 0.72 個百分點的下降，建議改用 7.55 GB 的 AWQ 版本。
