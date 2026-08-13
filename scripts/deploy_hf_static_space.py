"""Validate and deploy the free static Hugging Face portfolio Space."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_SOURCE = ROOT / "space_static"
SPACE_REPO = "steven0226/qwen3vl-chartqa-demo"
APP_URL = "https://steven0226-qwen3vl-chartqa-demo.static.hf.space/index.html"
REQUIRED_FILES = {
    "README.md",
    "index.html",
    "demo_gradio_colab.ipynb",
    "assets/latency_throughput.png",
    "assets/synthetic_ood_04.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="只驗證本機檔案，不建立或更新 Space"
    )
    parser.add_argument(
        "--wait", action="store_true", help="部署後等待公開頁面可正常讀取"
    )
    parser.add_argument("--timeout-minutes", type=int, default=10)
    return parser.parse_args()


def preflight() -> None:
    missing = sorted(
        path for path in REQUIRED_FILES if not (SPACE_SOURCE / path).is_file()
    )
    if missing:
        raise RuntimeError(f"Static Space 缺少必要檔案: {missing}")

    readme = (SPACE_SOURCE / "README.md").read_text(encoding="utf-8")
    index = (SPACE_SOURCE / "index.html").read_text(encoding="utf-8")
    checks = {
        "README sdk": "sdk: static" in readme,
        "README app_file": "app_file: index.html" in readme,
        "README valid color": "colorFrom: red" in readme,
        "HTML zh-Hant": 'lang="zh-Hant"' in index,
        "Colab notebook link": "demo_gradio_colab.ipynb" in index,
        "GGUF revision": "5e5860f5d406" in index,
        "AWQ result": "85.52%" in index,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Static Space preflight 失敗: {failed}")
    print(f"Static Space preflight PASS: {len(REQUIRED_FILES)} required files")


def wait_until_public(timeout_minutes: int) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    last_status: int | str = "尚未連線"
    with httpx.Client(follow_redirects=True, timeout=20) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(APP_URL)
                last_status = response.status_code
                if response.status_code == 200 and "問圖表" in response.text:
                    print(f"Static Space public PASS: {APP_URL}")
                    return
            except httpx.HTTPError as exc:
                last_status = type(exc).__name__
            print(f"等待 Static Space 發布：{last_status}")
            time.sleep(10)
    raise TimeoutError(
        f"Static Space 在 {timeout_minutes} 分鐘內尚未可讀；最後狀態: {last_status}"
    )


def main() -> None:
    args = parse_args()
    preflight()
    if args.check_only:
        return

    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("找不到 .env 的 HF_TOKEN")
    api = HfApi(token=token)
    username = api.whoami().get("name")
    if username != "steven0226":
        raise RuntimeError(f"HF_TOKEN 帳號不符: {username!r}")

    api.create_repo(
        SPACE_REPO,
        repo_type="space",
        space_sdk="static",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=SPACE_REPO,
        repo_type="space",
        folder_path=SPACE_SOURCE,
        commit_message="Publish static ChartQA portfolio",
        delete_patterns=["style.css"],
    )
    print(f"Space commit: {commit.oid}")
    print(f"Space page: https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"App URL: {APP_URL}")
    if args.wait:
        wait_until_public(args.timeout_minutes)


if __name__ == "__main__":
    main()
