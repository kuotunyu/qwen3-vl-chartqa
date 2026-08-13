---
title: Qwen3-VL ChartQA｜Live
emoji: 📊
colorFrom: orange
colorTo: yellow
sdk: docker
app_port: 7860
license: mit
models:
  - steven0226/qwen3vl-8b-chartqa-gguf
datasets:
  - HuggingFaceM4/ChartQA
preload_from_hub:
  - steven0226/qwen3vl-8b-chartqa-gguf Qwen3VL-8B-ChartQA-Q4_K_M.gguf,mmproj-Qwen3VL-8B-ChartQA-Q8_0.gguf __MODEL_REVISION__
---

# Qwen3-VL ChartQA｜頁面內直接推論

這是 `Qwen3-VL-8B-Instruct` 經 15,000 筆 ChartQA QLoRA fine-tuning 後的持久展示版。

- Runtime：fine-tuned GGUF `Q4_K_M` + `Q8_0 mmproj`
- Serving：pinned `llama.cpp`，Hugging Face Docker Space
- 正式部署模型與品質報告：[AWQ W4A16 g32](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq)
- 完整專案結果：AWQ 85.52%，量化掉分 0.72pp；A100 vLLM benchmark 8/8 組成功

GGUF 是低成本持久展示路線，發布前會做獨立 smoke verification；正式 2,500 題品質數字仍以 AWQ/vLLM 評估為準，不跨推論 stack 混用。

## Production controls

- 單一模型執行槽＋有限等待佇列，避免 CPU 記憶體與延遲失控
- 每 IP sliding-window rate limit、問題長度、圖片像素／payload 與輸出 token 上限
- llama.cpp 背景載入、bounded restart budget、graceful shutdown 與結構化 JSON log
- `GET /healthz`：process liveness
- `GET /readyz`：model readiness
- `GET /api/status`：queue、latency、success/error 與安全的模型識別資訊
- `GET /metrics`：Prometheus text format
- `POST /api/v1/infer`：供 OOD evaluation 與 soak test 共用的 JSON API
