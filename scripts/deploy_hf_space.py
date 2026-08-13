"""Validate the GGUF artifact and deploy the persistent Docker Space.

Run from the repository root after convert_gguf_colab_cpu_fixed.ipynb has completed:

    uv run python scripts/deploy_hf_space.py --check-only
    uv run python scripts/deploy_hf_space.py --wait

The script reads HF_TOKEN from .env, pins the model revision into the uploaded
Space README and Space variables, and never modifies the project source files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]
SPACE_SOURCE = ROOT / "space"
MODEL_REPO = "steven0226/qwen3vl-8b-chartqa-gguf"
DEFAULT_SPACE_REPO = "steven0226/qwen3vl-chartqa-live"
MODEL_FILE = "Qwen3VL-8B-ChartQA-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-Qwen3VL-8B-ChartQA-Q8_0.gguf"
REQUIRED_FILES = {MODEL_FILE, MMPROJ_FILE, "conversion_metadata.json", "README.md"}
TERMINAL_FAILURE_STAGES = {
    "NO_APP_FILE",
    "CONFIG_ERROR",
    "BUILD_ERROR",
    "RUNTIME_ERROR",
    "STOPPED",
    "PAUSED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="只驗證模型產物，不建立或更新 Space"
    )
    parser.add_argument(
        "--wait", action="store_true", help="部署後等待 Space RUNNING 或失敗"
    )
    parser.add_argument("--timeout-minutes", type=int, default=35)
    parser.add_argument(
        "--space-repo",
        default=DEFAULT_SPACE_REPO,
        help="Docker Space repo；預設使用獨立 live repo，避免破壞既有 Static Space",
    )
    return parser.parse_args()


def model_preflight(api: HfApi) -> tuple[str, dict[str, int]]:
    if not api.repo_exists(MODEL_REPO, repo_type="model"):
        raise RuntimeError(
            f"找不到 {MODEL_REPO}；請先在 Colab 全部執行 "
            "notebooks/convert_gguf_colab_cpu_fixed.ipynb。"
        )
    info = api.model_info(MODEL_REPO, files_metadata=True)
    sizes = {item.rfilename: int(item.size or 0) for item in info.siblings}
    missing = sorted(REQUIRED_FILES - sizes.keys())
    if missing:
        raise RuntimeError(f"GGUF repo 缺少必要檔案: {missing}")
    if not 4_500_000_000 <= sizes[MODEL_FILE] <= 6_000_000_000:
        raise RuntimeError(f"Q4_K_M 檔案大小異常: {sizes[MODEL_FILE]:,} bytes")
    if not 500_000_000 <= sizes[MMPROJ_FILE] <= 1_500_000_000:
        raise RuntimeError(f"Q8_0 mmproj 檔案大小異常: {sizes[MMPROJ_FILE]:,} bytes")
    return info.sha, sizes


def staged_space_source(model_revision: str) -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="qwen3vl-space-")
    dst = Path(tmp.name)
    for source in SPACE_SOURCE.iterdir():
        if source.name == "__pycache__":
            continue
        if source.is_file():
            shutil.copy2(source, dst / source.name)
    readme = dst / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "__MODEL_REVISION__"
    if text.count(marker) != 1:
        tmp.cleanup()
        raise RuntimeError(f"Space README 應恰好包含一個 {marker}")
    readme.write_text(
        text.replace(marker, model_revision), encoding="utf-8", newline="\n"
    )
    return tmp


def space_stage_value(stage: object) -> str:
    """Normalize huggingface_hub SpaceStage enums without version-specific strings."""
    return str(getattr(stage, "value", stage))


def app_url(space_repo: str) -> str:
    owner, name = space_repo.split("/", 1)
    return f"https://{owner}-{name}.hf.space"


def wait_until_ready(api: HfApi, space_repo: str, timeout_minutes: int) -> None:
    deadline = time.monotonic() + timeout_minutes * 60
    last_stage = None
    last_readiness = None
    url = app_url(space_repo)
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            runtime = api.get_space_runtime(space_repo)
            stage = space_stage_value(runtime.stage)
            if stage != last_stage:
                print("Space stage:", stage)
                last_stage = stage
            if stage in TERMINAL_FAILURE_STAGES:
                raise RuntimeError(f"Space build/runtime 失敗: {stage}")
            if stage == "RUNNING":
                try:
                    response = client.get(f"{url}/readyz")
                    payload = response.json()
                    readiness = (response.status_code, payload.get("state"))
                    if readiness != last_readiness:
                        print(
                            "Model readiness:",
                            response.status_code,
                            payload.get("state"),
                        )
                        last_readiness = readiness
                    if response.status_code == 200 and payload.get("ready") is True:
                        print("Docker Space public READY:", url)
                        return
                except (httpx.HTTPError, ValueError):
                    pass
            time.sleep(15)
    raise TimeoutError(
        f"Space 在 {timeout_minutes} 分鐘內未 ready；"
        f"最後 stage={last_stage}, readiness={last_readiness}"
    )


def main() -> None:
    args = parse_args()
    space_repo = args.space_repo
    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("找不到 .env 的 HF_TOKEN")
    api = HfApi(token=token)
    who = api.whoami()
    if who.get("name") != "steven0226":
        raise RuntimeError(f"HF_TOKEN 帳號不符: {who.get('name')!r}")

    revision, sizes = model_preflight(api)
    print(f"GGUF preflight PASS: {MODEL_REPO}@{revision}")
    print(f"  model:  {sizes[MODEL_FILE] / 1024**3:.2f} GiB")
    print(f"  mmproj: {sizes[MMPROJ_FILE] / 1024**3:.2f} GiB")
    if args.check_only:
        return

    try:
        api.create_repo(
            space_repo,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 402:
            raise RuntimeError(
                "Hugging Face 拒絕建立 Docker Space（HTTP 402）。"
                "目前 token 所屬帳號不是 PRO；請先在帳號 Billing/PRO 完成訂閱，"
                "再重新執行同一命令。既有 Static Space 未被修改。"
            ) from exc
        raise
    info = api.space_info(space_repo)
    if info.sdk != "docker":
        raise RuntimeError(
            f"{space_repo} SDK 是 {info.sdk!r}，不是 docker；為避免破壞既有 Space 已停止。"
        )
    variables = {
        "MODEL_REPO": MODEL_REPO,
        "MODEL_REVISION": revision,
        "MODEL_FILE": MODEL_FILE,
        "MMPROJ_FILE": MMPROJ_FILE,
        "MAX_QUEUE_SIZE": "6",
        "QUEUE_WAIT_TIMEOUT_S": "180",
        "RATE_LIMIT_REQUESTS": "12",
        "RATE_LIMIT_WINDOW_S": "600",
        "INFERENCE_TIMEOUT_S": "360",
        "MAX_RESTARTS_PER_HOUR": "3",
    }
    for key, value in variables.items():
        api.add_space_variable(space_repo, key=key, value=value)
    admin_token = os.getenv("SPACE_ADMIN_TOKEN")
    if admin_token:
        api.add_space_secret(space_repo, key="ADMIN_TOKEN", value=admin_token)

    staged = staged_space_source(revision)
    try:
        commit = api.upload_folder(
            repo_id=space_repo,
            repo_type="space",
            folder_path=staged.name,
            commit_message=f"Deploy pinned ChartQA GGUF {revision[:12]}",
        )
    finally:
        staged.cleanup()
    print("Space commit:", commit.oid)
    print(f"Space page: https://huggingface.co/spaces/{space_repo}")
    print("App URL:", app_url(space_repo))

    if not args.wait:
        return
    wait_until_ready(api, space_repo, args.timeout_minutes)


if __name__ == "__main__":
    main()
