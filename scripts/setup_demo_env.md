# Phase 5 demo 環境設定（Windows GPU 機器）

目標機器：RTX 4090 24GB（`lora_4bit` 模式約需 10GB VRAM；<12GB 的卡會 OOM）。

## 安裝

`pyproject.toml` 已透過 `[tool.uv.sources]` 把 Windows 上的 `torch`/`torchvision`
指到 PyTorch 官方 cu126 wheel index，因此只需要：

```powershell
uv sync --extra demo     # 第一次會下載 CUDA 版 torch（~2.7GB）
scripts/run_demo.ps1     # 預設 lora_4bit；-Mode dummy 可先驗 UI
```

驗證 CUDA 可用：

```powershell
uv run --extra demo python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 模式

| DEMO_MODE | 模型 | VRAM 估計 | 備註 |
|---|---|---|---|
| `lora_4bit`（預設） | `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` + `<HF_USER>/qwen3vl-8b-chartqa-lora` | ~10GB | 與 Phase 2/3 同組合；adapter 不存在時自動退回 base 並警告 |
| `awq` | `<HF_USER>/qwen3vl-8b-chartqa-awq` | ~10GB† | Phase 4 產物；transformers 載 compressed-tensors 屬實驗性，啟動時看 VRAM 實測 |
| `merged16` | `<HF_USER>/qwen3vl-8b-chartqa-merged-16bit` | ~18GB+ | 24GB 卡邊緣，僅實驗 |
| `dummy` | 無 | 0 | UI 管線測試，不需要 GPU |

† transformers 若走 decompress 路徑會膨脹回 bf16（~17.5GB）；啟動 log 的
`[VRAM model loaded]` 若超過 ~20GB 建議改回 `lora_4bit`。

## HF 帳號

adapter/量化 repo 的帳號來自 `.env` 的 `HF_TOKEN`（`whoami()`）；
要覆蓋時設環境變數 `HF_USER=<帳號名>`。
