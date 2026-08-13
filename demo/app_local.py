"""Phase 5 local Gradio demo: chart understanding with the fine-tuned Qwen3-VL-8B.

Modes (env var DEMO_MODE or --mode):
    lora_4bit  (default) pre-quantized bnb-4bit base + LoRA adapter from the Hub —
               the exact combo Phase 2 trained and Phase 3 measured (~10GB VRAM)
    awq        Phase 4 AWQ W4A16 repo (compressed-tensors; experimental in transformers)
    merged16   merged bf16 repo (~18GB+ VRAM, experimental)
    dummy      no model, no GPU — UI wiring test only

Run (Windows, from the repo root):
    scripts/run_demo.ps1                 # defaults to lora_4bit
    scripts/run_demo.ps1 -Mode dummy     # UI test without a GPU
CUDA torch setup for the GPU machine: see scripts/setup_demo_env.md
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from src.chartqa_data import ANSWER_INSTRUCTION  # noqa: E402

SHORT_ANSWER = "ChartQA 短答"
FREE_FORM = "自由問答"


def resolve_user() -> str:
    user = os.environ.get("HF_USER")
    if user:
        return user
    from huggingface_hub import whoami

    return whoami()["name"]


def vram_report(tag: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[VRAM {tag}] allocated={a:.1f}GB reserved={r:.1f}GB / total {total:.0f}GB")


def load_model(mode: str):
    """Returns (model, processor); (None, None) in dummy mode."""
    if mode == "dummy":
        print("[dummy] 不載入模型，僅測試 UI 管線")
        return None, None

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        sys.exit(
            "需要 CUDA GPU 才能跑模型（CUDA 版 torch 安裝見 scripts/setup_demo_env.md；"
            "純 UI 測試請用 --mode dummy）"
        )
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total < 12:
        print(
            f"!! 警告：VRAM {total:.0f}GB < 12GB，8B 模型很可能 OOM（demo 目標機器是 4090 24GB）"
        )
    # bf16 需要 Ampere(8.0)+；更舊的卡退 fp16
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(0) >= (8, 0)
        else torch.float16
    )

    user = resolve_user()
    if mode == "lora_4bit":
        # 與 Phase 2/3 同一組合：nf4 預量化底模 + LoRA adapter，行為對得上已量測分數
        base = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base, device_map="cuda:0", dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(base)
        adapter = f"{user}/qwen3vl-8b-chartqa-lora"
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            print(f"adapter 已掛載: {adapter}")
        except Exception as e:
            print(f"!! 掛不到 adapter（{adapter}）: {type(e).__name__}: {e}")
            print("!! 先以未微調的 base model 執行；Phase 2 訓練推上 Hub 後重啟即可")
    elif mode == "awq":
        repo = f"{user}/qwen3vl-8b-chartqa-awq"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo, device_map="cuda:0", dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(repo)
    elif mode == "merged16":
        repo = f"{user}/qwen3vl-8b-chartqa-merged-16bit"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo, device_map="cuda:0", dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(repo)
    else:
        raise ValueError(f"unknown mode: {mode}")

    model.eval()
    vram_report("model loaded")
    return model, processor


def build_answer_fn(model, processor):
    from transformers import TextIteratorStreamer

    def answer(image, question, mode, max_new_tokens):
        if image is None or not (question or "").strip():
            yield "請先上傳圖表並輸入問題"
            return
        text = question if mode == FREE_FORM else f"{question}\n{ANSWER_INSTRUCTION}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[prompt], images=[image.convert("RGB")], return_tensors="pt"
        ).to("cuda")
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
        thread = threading.Thread(target=model.generate, kwargs=kwargs)
        thread.start()
        out = ""
        for piece in streamer:
            out += piece
            yield out
        thread.join()
        vram_report("after generate")

    return answer


def dummy_answer(image, question, mode, max_new_tokens):
    yield (
        f"[dummy 模式] 收到問題: {question!r}\n"
        f"image={'有' if image is not None else '無'}  mode={mode}  "
        f"max_new_tokens={int(max_new_tokens)}\n"
        f"（短答模式會附加指令: {ANSWER_INSTRUCTION!r}）"
    )


def build_ui(answer_fn, mode_tag: str):
    import gradio as gr

    with gr.Blocks(title="Qwen3-VL-8B ChartQA demo") as ui:
        gr.Markdown(
            f"# 圖表理解 demo — Qwen3-VL-8B ChartQA QLoRA\n"
            f"模式：`{mode_tag}`。上傳圖表、輸入問題；「{SHORT_ANSWER}」比照評估用的"
            f"短答指令，「{FREE_FORM}」自由發揮（建議調高 max_new_tokens）。"
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="圖表")
                question = gr.Textbox(
                    label="問題", placeholder="例：Which year has the highest value?"
                )
                mode = gr.Radio(
                    [SHORT_ANSWER, FREE_FORM], value=SHORT_ANSWER, label="回答模式"
                )
                max_new = gr.Slider(
                    8, 512, value=32, step=8, label="max_new_tokens（短答 32 即可）"
                )
                btn = gr.Button("提問", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="回答", lines=10)
        btn.click(answer_fn, [image, question, mode, max_new], out)
        question.submit(answer_fn, [image, question, mode, max_new], out)
    return ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("DEMO_MODE", "lora_4bit"),
        choices=["lora_4bit", "awq", "merged16", "dummy"],
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model, processor = load_model(args.mode)
    answer_fn = dummy_answer if model is None else build_answer_fn(model, processor)
    ui = build_ui(answer_fn, args.mode)
    ui.launch(server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
