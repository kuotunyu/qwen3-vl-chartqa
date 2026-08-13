# Phase 5 demo launcher.
#   scripts/run_demo.ps1                 # lora_4bit（bnb-4bit 底模 + LoRA adapter）
#   scripts/run_demo.ps1 -Mode awq       # Phase 4 量化版
#   scripts/run_demo.ps1 -Mode dummy     # 不載模型，純 UI 測試
# CUDA torch 安裝見 scripts/setup_demo_env.md
param(
    [ValidateSet("lora_4bit", "awq", "merged16", "dummy")]
    [string]$Mode = "lora_4bit",
    [int]$Port = 7860
)
$env:DEMO_MODE = $Mode
Set-Location (Split-Path $PSScriptRoot -Parent)
uv run --extra demo python demo/app_local.py --mode $Mode --port $Port
