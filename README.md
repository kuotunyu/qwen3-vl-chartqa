# Qwen3-VL ChartQA：QLoRA 微調、AWQ 量化與 vLLM 部署

[![CI](https://github.com/kuotunyu/qwen3-vl-chartqa/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/qwen3-vl-chartqa/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/kuotunyu/qwen3-vl-chartqa/releases/tag/v1.0.0)
[![Status](https://img.shields.io/badge/Status-Complete%20%2F%20Portfolio%20Flagship-success.svg)](https://github.com/kuotunyu/qwen3-vl-chartqa/releases/tag/v1.0.0)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![vLLM](https://img.shields.io/badge/vLLM-0.25.1-6366F1)
![AWQ](https://img.shields.io/badge/Quantization-AWQ%20W4A16-009688)
[![License: MIT](https://img.shields.io/badge/License-MIT-2EA44F.svg)](LICENSE)

[English](README.en.md)

本專案為針對圖表問答（ChartQA）任務打造的視覺語言模型（VLM）端到端工程：以 15,000 筆資料微調 Qwen3-VL-8B-Instruct，在 2,500 題完整 Test 集上驗證 AWQ W4A16 量化品質（僅下降 0.72 pp，通過 2 pp 品質門檻），並於單張 NVIDIA A100 上完成 cache-controlled 的 vLLM Serving Benchmark（吞吐量提升 83.2%，TPOT p95 降低 51.9%）。

### 30 秒快速摘要 (Executive Summary)

| 階段 (Stage) | 核心指標與成果 (Key Result) | 工程控制與產物 (Controls & Artifacts) |
|---|---|---|
| **QLoRA 微調** | 完整 2,500 題 ChartQA Test：**+0.56 pp**（84.68% → 85.24%） | 15,000 樣本、全適配 Vision/Language，生成 LoRA 與 Merged 16-bit 權重 |
| **AWQ 量化門禁** | AWQ W4A16 僅下降 **-0.72 pp**，通過預設 **-2.0 pp** 嚴格品質門檻 | 權重由 17.53 GB 降至 7.55 GB（**2.32× 壓縮**，減少 56.9%） |
| **vLLM A100 部署** | 輸出吞吐量最高提升 **+83.2%**，TPOT p95 最高降低 **-51.9%** | 單張 A100、關閉快顯快取、嚴格分離 Warmup 與量測窗、8 組併發 100% 成功 |
| **多格式產物庫** | 公開 4 種權重格式 + 靜態展示頁 + Colab A100 互動環境 | LoRA / Merged 16-bit / AWQ / GGUF（llama.cpp CPU 驗證） |
| **自動化審計門禁** | **266 條機械驗證全部通過**（`python scripts/verify_claims.py`） | 數據與報告直接由 `assets/` 證據鏈重算，無任何宣稱造假 |

---

## 系統架構與 Pipeline

### 1. 視覺語言模型端到端生命週期 Pipeline

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph TrainStage ["階段一：資料工程與 QLoRA 微調 (Data & QLoRA Fine-tuning)"]
        direction LR
        Data[("ChartQA 訓練資料集<br/>(15,000 筆隨機抽樣)")] --> QLoRA["Qwen3-VL-8B QLoRA 微調<br/>(Vision + Language 模組全適配)"] --> Merge[("Merged 16-bit 權重<br/>(完整精度基準模型)")]
    end

    subgraph QuantStage ["階段二：AWQ 量化與品質門檻 (Quantization & Quality Gate)"]
        direction LR
        Merge --> AWQ["AWQ W4A16 g32 量化<br/>(256 筆 Calibration 樣本)"] --> Gate{"品質門檻檢驗<br/>(允許下降 ≤ 2.0 pp)"} --> Passed[("AWQ 正式發布產物<br/>(7.55 GB · 2.32× 壓縮)")]
    end

    subgraph ServeStage ["階段三：vLLM 高併發部署與展示 (vLLM Serving & Demo)"]
        direction LR
        Passed --> vLLM["vLLM A100 Serving 引擎<br/>(快取嚴格隔離 · 123.2 tok/s)"] --> Demo(["Gradio / Space 互動展示<br/>(Colab A100 一鍵執行)"])
    end

    TrainStage --> QuantStage --> ServeStage

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef condStyle fill:#fff9db,stroke:#f59f00,stroke-width:2px,color:#212529
    classDef evalStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class Data,Merge,Passed srcStyle
    class QLoRA,AWQ,vLLM procStyle
    class Gate condStyle
    class Demo evalStyle

    style TrainStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style QuantStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style ServeStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

### 2. 服務部署與多端推論架構

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph ArtStage ["階段一：多格式模型產物庫 (Model Artifacts)"]
        direction LR
        M1[("AWQ W4A16 g32<br/>(建議部署版本 · 7.55 GB)")]
        M2[("Merged 16-bit<br/>(品質基準 · 17.53 GB)")]
        M3[("GGUF Q4_K_M + Q8_0<br/>(llama.cpp CPU 可攜式)")]
    end

    subgraph EngineStage ["階段二：多端推論與快取控制 (Inference Engines)"]
        direction LR
        vLLMEng["vLLM SXM4-A100 伺服引擎<br/>(關閉 Processor/Prefix Cache)"]
        CPUEng["llama.cpp CPU 離線推論<br/>(獨立 Smoke Test 驗證)"]
    end

    subgraph DeliveryStage ["階段三：成果展示與客觀審計 (Delivery & Verification)"]
        direction LR
        Space(["Hugging Face Static Space<br/>(免權重靜態作品展示頁)"])
        Colab(["Colab A100 互動 Notebook<br/>(一鍵載入 AWQ 即時問答)"])
        Audit{"266 條機械驗證門禁<br/>(verify_claims.py)"}
    end

    M1 & M2 --> vLLMEng
    M3 --> CPUEng
    vLLMEng --> Colab
    CPUEng --> Audit
    vLLMEng --> Space

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef condStyle fill:#fff9db,stroke:#f59f00,stroke-width:2px,color:#212529
    classDef evalStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class M1,M2,M3 srcStyle
    class vLLMEng,CPUEng procStyle
    class Audit condStyle
    class Space,Colab evalStyle

    style ArtStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style EngineStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style DeliveryStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

---

## 模型產物

| 產物 | 用途 | 規格與權重連結 |
|---|---|---|
| AWQ W4A16 g32 | 建議部署版本（2.32× 壓縮、高吞吐） | [Hugging Face 權重](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq) |
| Merged 16-bit | 品質參考與 full-precision baseline | [Hugging Face 權重](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-merged-16bit) |
| LoRA adapter | 訓練權重與 PEFT 模組 | [Hugging Face 權重](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-lora) |
| GGUF Q4_K_M + Q8_0 mmproj | llama.cpp CPU 可攜式推論產物 | [Hugging Face 權重](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-gguf) |

線上展示：[Hugging Face Space 作品頁](https://huggingface.co/spaces/steven0226/qwen3vl-chartqa-demo)（支援 Colab A100 一鍵啟動）

---

## 正式評測結果

以下所有數據均可透過 `python scripts/verify_claims.py` 由 `assets/` 內的不可變證據直接重算驗證：

### 微調前後

完整 2,500 題 ChartQA test set 配對評估：

| Split | n | 微調前 | 微調後 | 變化 |
|---|---:|---:|---:|---:|
| Human | 1,250 | 75.28% | 75.44% | +0.16 pp |
| Augmented | 1,250 | 94.08% | 95.04% | +0.96 pp |
| Overall | 2,500 | 84.68% | **85.24%** | **+0.56 pp** |

### 量化品質門檻

Merged 16-bit 與 AWQ 於隔離 vLLM 程序中進行 2,500 題完整配對評估（預設容許上限：-2.0 pp）：

| Split | n | Merged 16-bit | AWQ W4A16 g32 | 變化 |
|---|---:|---:|---:|---:|
| Human | 1,250 | 77.28% | 76.56% | -0.72 pp |
| Augmented | 1,250 | 95.20% | 94.48% | -0.72 pp |
| Overall | 2,500 | 86.24% | **85.52%** | **-0.72 pp — PASS** |

整體 AWQ 變化的 paired bootstrap 95% CI 為 `[-1.40, -0.04] pp`，完全落在 -2.0 pp 品質門檻內。

### Serving benchmark

基準測試環境為單張 NVIDIA A100-SXM4-40GB、vLLM `0.25.1+cu129`；各 level 包含 64 筆正式請求與固定 64 tokens 解碼，8 組測試全部 64/64 成功並通過 validity gate：

| 模型 | Concurrency | Output tok/s | TTFT p95 | TPOT p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|
| Merged 16-bit | 1 | 67.29 | 160.76 ms | 13.43 ms/tok | 1,007.16 ms |
| AWQ W4A16 g32 | 1 | **123.24** | **154.81 ms** | **6.47 ms/tok** | **562.00 ms** |
| Merged 16-bit | 4 | 231.02 | **299.78 ms** | 15.04 ms/tok | 1,169.61 ms |
| AWQ W4A16 g32 | 4 | **356.95** | 326.64 ms | **9.11 ms/tok** | **776.13 ms** |
| Merged 16-bit | 8 | 387.58 | **473.68 ms** | 18.33 ms/tok | 1,426.90 ms |
| AWQ W4A16 g32 | 8 | **528.48** | 572.66 ms | **12.40 ms/tok** | **1,038.69 ms** |
| Merged 16-bit | 16 | 595.06 | **806.77 ms** | 24.14 ms/tok | 2,005.45 ms |
| AWQ W4A16 g32 | 16 | **701.90** | 958.28 ms | **19.93 ms/tok** | **1,612.51 ms** |

![A100 vLLM latency and throughput benchmark](assets/bench/latency_throughput.png)

相較於 Merged 16-bit，AWQ 展現顯著部署效益：

- 權重檔由 17.53 GB 降至 7.55 GB，減少 56.9%（2.32× 壓縮）；
- 在 concurrency 1/4/8/16 的輸出吞吐量分別提升 83.2%／54.5%／36.4%／18.0%；
- TPOT p95 分別降低 51.9%／39.4%／32.4%／17.4%；
- E2E p95 分別降低 44.2%／33.6%／27.2%／19.6%；
- 高併發 TTFT p95（c=4/8/16）增加 9.0%／20.9%／18.8%，屬合理計算取捨。

---

## 方法與工程控制

- **QLoRA 微調：** 使用 8-bit AdamW、peak lr `2e-4`、effective batch size 16，在 A100 上對 Qwen3-VL-8B 視覺與語言模組進行全適配微調（3,579 秒收斂，train loss 0.5907）。
- **AWQ W4A16 量化：** group size 32，使用 256 筆校準樣本，保留 vision tower 與 `lm_head` 原始精度。
- **GGUF 匯出：** 生成 `Q4_K_M` 文字模型與 `Q8_0` 多模態 projector，通過獨立 CPU smoke test。
- **嚴格 Benchmark 控制：** 關閉快顯緩存、嚴格分離 warmup/measured 圖片、綁定單一物理 GPU，量測期若有 JIT 立即駁回。

---

## 驗證與重現

本機離線驗證（不需 GPU、不需權重與資料集）：

```bash
uv sync --python 3.12
uv run python scripts/verify_claims.py
uv run python -m unittest discover -s tests -v
```

---

## 資料與授權

- 本專案程式碼採 [MIT License](LICENSE) 授權。
- 第三方元件與資料集授權詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
