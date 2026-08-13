---
license: apache-2.0
base_model: unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: image-text-to-text
tags:
- qwen3-vl
- vision-language
- chartqa
- qlora
- peft
- unsloth
- trl
datasets:
- HuggingFaceM4/ChartQA
language:
- en
---

# Qwen3-VL-8B ChartQA — LoRA Adapter

QLoRA adapter fine-tuned for short-answer chart question answering on ChartQA. This repository contains the adapter and training/evaluation artifacts; use the [merged 16-bit](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-merged-16bit) or [AWQ W4A16](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq) repository for standalone serving.

## Training

- Base: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`
- Data: 15,000 shuffled examples from `HuggingFaceM4/ChartQA` train
- Epochs: 1
- Maximum sequence length: 2,048
- LoRA rank/alpha/dropout: 16 / 16 / 0
- Vision, language, attention, and MLP modules enabled
- Effective A100 batch size: 16 (4 per device × 4 gradient accumulation)
- Optimizer: 8-bit AdamW; peak learning rate `2e-4`; linear schedule
- Seed: 3407
- Runtime: 3,579 seconds
- Final training loss: 0.5907

![Training loss](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-lora/resolve/main/loss_curve.png)

## Paired evaluation

Complete ChartQA test set; relaxed accuracy with 5% numeric tolerance:

| Split | n | Base before | Fine-tuned after | Change |
|---|---:|---:|---:|---:|
| Human | 1,250 | 75.28% | 75.44% | +0.16 pp |
| Augmented | 1,250 | 94.08% | 95.04% | +0.96 pp |
| Overall | 2,500 | 84.68% | **85.24%** | **+0.56 pp** |

The gain is concentrated on augmented questions; improvement on human-authored questions is small. Aggregate results and content-free per-item correctness evidence are published in the [GitHub repository](https://github.com/kuotunyu/qwen3-vl-chartqa); raw predictions, questions, labels, and ChartQA images are intentionally excluded from public artifacts.

## Loading with Unsloth

Use a current Unsloth environment with Qwen3-VL support:

```python
from unsloth import FastVisionModel

model, processor = FastVisionModel.from_pretrained(
    "steven0226/qwen3vl-8b-chartqa-lora",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
```

Format inputs as a Qwen3-VL user message containing the chart image followed by the question and:

```text
Answer the question using a single word or phrase.
```

For deployment without PEFT attachment, use the merged or AWQ repositories linked above.

## Intended use

- Research and portfolio demonstrations of QLoRA for chart understanding
- Continued fine-tuning or adapter-based experimentation
- Reproducing the paired ChartQA evaluation

## Limitations

- Evaluated primarily on English ChartQA short answers.
- The overall improvement is modest (+0.56 pp), and the human subset gain is only +0.16 pp.
- Not validated for arbitrary documents, OCR-heavy scans, or safety-critical decisions.
- The adapter depends on the recorded base model and compatible Qwen3-VL/Unsloth versions.

## 中文摘要

這是以 15,000 筆 ChartQA、1 epoch 訓練的 Qwen3-VL-8B QLoRA adapter。完整 2,500 題 relaxed accuracy 由 84.68% 提升至 85.24%（+0.56 pp）；增益主要來自 augmented split，human split 僅提升 0.16 pp。若要直接部署，建議使用已合併的 16-bit 或 AWQ W4A16 版本。
