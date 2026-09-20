# 服務部署與多端推論架構 / Multi-Target Serving Architecture

這張圖原本放在 README，說明公開的多格式權重如何對應到各推論引擎與展示方式。端到端的訓練→量化→部署流程圖留在 [README](../README.md#運作方式)。

This diagram used to live in the README. It shows how the published weight formats map onto the inference engines and showcase pages. The end-to-end train → quantize → serve pipeline stays in the [README](../README.en.md#how-it-works).

## 服務部署與多端推論架構

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
        Audit{"離線證據一致性檢查<br/>(verify_claims.py)"}
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

## Multi-Target Serving Architecture

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph ArtStage ["Phase 1: Multi-Format Model Artifacts"]
        direction LR
        M1[("AWQ W4A16 g32<br/>(Recommended release · 7.55 GB)")]
        M2[("Merged 16-bit<br/>(Quality baseline · 17.53 GB)")]
        M3[("GGUF Q4_K_M + Q8_0<br/>(llama.cpp CPU portable)")]
    end

    subgraph EngineStage ["Phase 2: Multi-Target Inference Engines"]
        direction LR
        vLLMEng["vLLM SXM4-A100 Serving<br/>(Processor & Prefix Cache disabled)"]
        CPUEng["llama.cpp CPU Offline Inference<br/>(Independent smoke test passed)"]
    end

    subgraph DeliveryStage ["Phase 3: Delivery & Verification Gates"]
        direction LR
        Space(["Hugging Face Static Space<br/>(Evidence-only static showcase)"])
        Colab(["Colab A100 Interactive Notebook<br/>(One-click AWQ live serving)"])
        Audit{"Offline Evidence Consistency Checks<br/>(verify_claims.py)"}
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
