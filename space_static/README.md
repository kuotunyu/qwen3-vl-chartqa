---
title: Qwen3-VL ChartQA｜圖表理解工作台
emoji: 📊
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
fullWidth: true
header: mini
pinned: true
license: mit
short_description: Qwen3-VL 8B 圖表理解微調、量化與 serving 實證作品
models:
  - steven0226/qwen3vl-8b-chartqa-lora
  - steven0226/qwen3vl-8b-chartqa-merged-16bit
  - steven0226/qwen3vl-8b-chartqa-awq
  - steven0226/qwen3vl-8b-chartqa-gguf
datasets:
  - HuggingFaceM4/ChartQA
tags:
  - vision-language-model
  - chart-question-answering
  - qwen3-vl
  - qlora
  - awq
  - gguf
  - vllm
---

# Qwen3-VL ChartQA｜圖表理解工作台

這是本專案的免費持久作品頁。Static Space 負責呈現可驗證成果；8B 模型推論透過頁面中的 Colab A100 notebook 啟動。

- 完整 ChartQA test：AWQ 85.52%，相對 merged 16-bit 下降 0.72 pp，通過 2 pp 品質門檻。
- A100 vLLM：8/8 組 benchmark 成功，concurrency 1 輸出吞吐提升 83.2%。
- fine-tuned GGUF：`Q4_K_M` + `Q8_0 mmproj`，獨立 CPU 圖表 smoke test 通過。
