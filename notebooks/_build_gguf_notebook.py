"""Maintenance source for the reviewed GGUF conversion Colab notebook.

Run from the repository root when the notebook must be regenerated:
    uv run --with nbformat==5.10.4 python notebooks/_build_gguf_notebook.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"gpuType": "A100", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    md(
        r"""
        # Qwen3-VL ChartQA — Fine-tuned GGUF 轉換與驗證（CPU-FIX v3）

        這份 notebook 只做 Phase 6 的一次性工作：把已驗證的 pinned merged-16bit 模型轉成可在免費 CPU Space 執行的 `Q4_K_M` GGUF，並輸出獨立的 `Q8_0 mmproj`。

        正確使用方式：

        1. Colab 執行階段選 **A100 GPU**，並允許 notebook 存取 Secret `HF_TOKEN`。
        2. 直接按「**全部執行**」，不要修改 repo、revision、llama.cpp commit 或檔名。
        3. 需要下載約 17.5GB merged 權重，另產生約 16.4GB 暫存 F16 GGUF；請使用全新 runtime。
        4. notebook 會以與最終 CPU Space 相同的 llama.cpp CPU backend，對執行期取得的 pinned ChartQA sample 做 smoke verification；公開產物不記錄題目、標籤或輸出。
        5. 成功後會建立／更新 `steven0226/qwen3vl-8b-chartqa-gguf`，並輸出最終 revision。

        為避免把不同 inference stack 的數字混用，本 notebook **不宣稱 GGUF 等於正式 AWQ 85.52%**。正式品質報告仍以 AWQ/vLLM 的完整 2,500 題評估為準；GGUF 是低成本持久展示路線。
        """
    ),
    code(
        r"""
        # 1. 全新 runtime、A100、RAM 與磁碟檢查
        import os, shutil, subprocess

        NOTEBOOK_BUILD_ID = "gguf-cpu-fixed-v3-20260716"
        print("GGUF_NOTEBOOK_BUILD:", NOTEBOOK_BUILD_ID)

        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        disk_gb = shutil.disk_usage("/content").free / 1024**3
        print("GPU:", gpu)
        print(f"System RAM: {ram_gb:.1f} GB")
        print(f"Disk free: {disk_gb:.1f} GB")
        assert "A100" in gpu, "請切換為 A100 GPU 後重新全部執行。"
        assert ram_gb >= 25, "系統 RAM 至少需要 25GB；請使用全新 A100 runtime。"
        assert disk_gb >= 55, "可用磁碟至少需要 55GB；請中斷並刪除舊 runtime 後重開。"
        """
    ),
    code(
        r"""
        # 2. 安裝轉換工具並編譯 pinned llama.cpp CPU 目標
        # 最終 Hugging Face Space 也是 CPU；不編譯與產物無關的巨型 CUDA template kernels。
        import os, signal, subprocess, sys, time
        from pathlib import Path

        LLAMA_CPP_COMMIT = "79bba02a6741de194912d370015866414faa83ad"
        LLAMA_DIR = Path("/content/llama.cpp")
        LOG_DIR = Path("/content/gguf_logs")
        LOG_DIR.mkdir(exist_ok=True)

        def tail(path, lines=80):
            path = Path(path)
            if not path.exists():
                return "(log 尚未建立)"
            return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

        def terminate_group(proc):
            if proc.poll() is not None:
                return
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)

        def run_heartbeat(cmd, log_name, label, cwd=None, env=None, stall_timeout_s=None):
            log_path = LOG_DIR / log_name
            print(f"[{label}] 開始；詳細 log -> {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [str(x) for x in cmd], cwd=cwd, env=env,
                    stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                )
                started = time.monotonic()
                last_progress = started
                last_size = 0
                next_heartbeat = 60
                try:
                    while proc.poll() is None:
                        now = time.monotonic()
                        elapsed = int(now - started)
                        size = log_path.stat().st_size if log_path.exists() else 0
                        if size != last_size:
                            last_size = size
                            last_progress = now
                        if elapsed >= next_heartbeat:
                            quiet_min = int((now - last_progress) // 60)
                            print(
                                f"[{label}] still running: {elapsed // 60} min | "
                                f"log {size / 1024**2:.1f} MB | last output {quiet_min} min ago",
                                flush=True,
                            )
                            next_heartbeat += 60
                        if stall_timeout_s and now - last_progress > stall_timeout_s:
                            terminate_group(proc)
                            print(tail(log_path))
                            raise RuntimeError(
                                f"{label} 超過 {stall_timeout_s // 60} 分鐘沒有新 log，已自動停止。"
                            )
                        time.sleep(2)
                except KeyboardInterrupt:
                    terminate_group(proc)
                    raise
            if proc.returncode != 0:
                print(tail(log_path))
                raise RuntimeError(f"{label} 失敗（exit={proc.returncode}）")
            print(tail(log_path, 20))
            return log_path

        run_heartbeat(
            ["apt-get", "update", "-qq"], "apt_update.log", "apt update",
        )
        run_heartbeat(
            ["apt-get", "install", "-y", "-qq", "build-essential", "cmake", "ninja-build", "git"],
            "apt_install.log", "apt install",
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)
        subprocess.run([
            "uv", "pip", "install", "--system", "--no-cache",
            "huggingface-hub==1.23.0", "datasets==5.0.0", "pillow==11.3.0",
        ], check=True)

        if LLAMA_DIR.exists():
            current = subprocess.run(
                ["git", "-C", str(LLAMA_DIR), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            assert current == LLAMA_CPP_COMMIT, (
                "現有 /content/llama.cpp 不是本專案 pinned commit；請重開全新 runtime。", current
            )
        else:
            run_heartbeat(
                ["git", "clone", "--filter=blob:none", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_DIR)],
                "git_clone.log", "clone llama.cpp",
            )
            subprocess.run(["git", "-C", str(LLAMA_DIR), "checkout", LLAMA_CPP_COMMIT], check=True)

        BUILD_JOBS = min(8, max(2, os.cpu_count() or 2))
        build_env = os.environ.copy()
        build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(BUILD_JOBS)
        print(f"llama.cpp CPU build jobs: {BUILD_JOBS}")
        run_heartbeat([
            "cmake", "-S", str(LLAMA_DIR), "-B", str(LLAMA_DIR / "build"), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CUDA=OFF", "-DGGML_NATIVE=OFF", "-DMTMD_VIDEO=OFF",
            "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=OFF",
        ], "cmake_configure.log", "configure llama.cpp", env=build_env)
        run_heartbeat([
            "cmake", "--build", str(LLAMA_DIR / "build"), "--config", "Release", f"-j{BUILD_JOBS}",
            "--target", "llama-quantize", "llama-mtmd-cli",
        ], "cmake_build.log", "build llama.cpp CPU", env=build_env, stall_timeout_s=20 * 60)

        QUANTIZE_BIN = LLAMA_DIR / "build/bin/llama-quantize"
        MTMD_BIN = LLAMA_DIR / "build/bin/llama-mtmd-cli"
        for binary in (QUANTIZE_BIN, MTMD_BIN):
            assert binary.exists(), f"缺少 binary: {binary}"
        print("llama.cpp pinned commit:", LLAMA_CPP_COMMIT)
        """
    ),
    code(
        r"""
        # 3. HF 認證、pinned merged 模型身分驗證與下載
        import json, os, time
        from pathlib import Path

        from google.colab import userdata
        from huggingface_hub import HfApi, snapshot_download

        HF_TOKEN = userdata.get("HF_TOKEN")
        assert HF_TOKEN, "找不到 Colab Secret: HF_TOKEN；請允許這份 notebook 存取。"
        os.environ["HF_TOKEN"] = HF_TOKEN
        HF_USER = "steven0226"
        MERGED_REPO = f"{HF_USER}/qwen3vl-8b-chartqa-merged-16bit"
        MERGED_REVISION = "519060ef43df3261e0512e5ae4c82a4d4e675f32"
        GGUF_REPO = f"{HF_USER}/qwen3vl-8b-chartqa-gguf"
        DATASET_ID = "HuggingFaceM4/ChartQA"
        DATASET_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
        MERGED_DIR = Path("/content/merged")
        OUT_DIR = Path("/content/gguf_out")
        OUT_DIR.mkdir(exist_ok=True)

        api = HfApi(token=HF_TOKEN)

        def retry(label, fn, attempts=6):
            for attempt in range(1, attempts + 1):
                try:
                    return fn()
                except Exception as exc:
                    if attempt == attempts:
                        raise
                    delay = min(60, 5 * 2 ** (attempt - 1))
                    print(f"{label} 暫時失敗（{type(exc).__name__}）；{delay}s 後重試 {attempt}/{attempts}")
                    time.sleep(delay)

        who = retry("HF whoami", api.whoami)
        assert who["name"] == HF_USER, who["name"]
        info = retry("merged model_info", lambda: api.model_info(
            MERGED_REPO, revision=MERGED_REVISION, files_metadata=True
        ))
        assert info.sha == MERGED_REVISION, (info.sha, MERGED_REVISION)
        weight_bytes = sum(int(item.size or 0) for item in info.siblings if item.rfilename.endswith(".safetensors"))
        assert 17_000_000_000 <= weight_bytes <= 18_500_000_000, weight_bytes
        print(f"Pinned merged weights: {weight_bytes / 1024**3:.2f} GiB")

        snapshot = retry("snapshot_download", lambda: snapshot_download(
            MERGED_REPO,
            revision=MERGED_REVISION,
            token=HF_TOKEN,
            local_dir=MERGED_DIR,
            allow_patterns=["*.json", "*.jinja", "*.model", "*.txt", "*.safetensors", "*.py"],
        ))
        config = json.loads((MERGED_DIR / "config.json").read_text(encoding="utf-8"))
        assert "Qwen3VLForConditionalGeneration" in config.get("architectures", []), config.get("architectures")
        assert len(list(MERGED_DIR.glob("*.safetensors"))) >= 4, "merged 權重分片不完整"
        print("Pinned merged snapshot ready:", snapshot)
        """
    ),
    code(
        r"""
        # 4. 轉換 text F16 + mmproj Q8_0，將 text 量化為 Q4_K_M
        import datetime as dt, hashlib, json, shutil
        from pathlib import Path

        CONVERT = LLAMA_DIR / "convert_hf_to_gguf.py"
        F16_PATH = OUT_DIR / "Qwen3VL-8B-ChartQA-F16.gguf"
        Q4_PATH = OUT_DIR / "Qwen3VL-8B-ChartQA-Q4_K_M.gguf"
        MMPROJ_PATH = OUT_DIR / "mmproj-Qwen3VL-8B-ChartQA-Q8_0.gguf"

        if not F16_PATH.exists() and not Q4_PATH.exists():
            run_heartbeat([
                sys.executable, str(CONVERT), str(MERGED_DIR),
                "--outfile", str(F16_PATH), "--outtype", "f16",
            ], "convert_text_f16.log", "convert text F16")
        if not MMPROJ_PATH.exists():
            run_heartbeat([
                sys.executable, str(CONVERT), str(MERGED_DIR), "--mmproj",
                "--outfile", str(MMPROJ_PATH), "--outtype", "q8_0",
            ], "convert_mmproj_q8.log", "convert mmproj Q8_0")
        if not Q4_PATH.exists():
            assert F16_PATH.exists(), "缺少 F16 中間檔"
            run_heartbeat([
                str(QUANTIZE_BIN), str(F16_PATH), str(Q4_PATH), "Q4_K_M", "8",
            ], "quantize_q4_k_m.log", "quantize Q4_K_M")

        assert 4_500_000_000 <= Q4_PATH.stat().st_size <= 6_000_000_000, Q4_PATH.stat().st_size
        assert 500_000_000 <= MMPROJ_PATH.stat().st_size <= 1_500_000_000, MMPROJ_PATH.stat().st_size
        if F16_PATH.exists():
            F16_PATH.unlink()
        assert shutil.disk_usage("/content").free / 1024**3 >= 8, "剩餘磁碟不足，請勿繼續上傳。"

        def sha256(path):
            h = hashlib.sha256()
            with Path(path).open("rb") as f:
                for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        metadata = {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_repo": MERGED_REPO,
            "source_revision": MERGED_REVISION,
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "quantization": "Q4_K_M",
            "mmproj_quantization": "Q8_0",
            "files": {
                Q4_PATH.name: {"bytes": Q4_PATH.stat().st_size, "sha256": sha256(Q4_PATH)},
                MMPROJ_PATH.name: {"bytes": MMPROJ_PATH.stat().st_size, "sha256": sha256(MMPROJ_PATH)},
            },
            "full_chartqa_evaluation": False,
            "formal_quality_reference": "steven0226/qwen3vl-8b-chartqa-awq@43b71926a1d645133560347787539729bcd3de6b",
        }
        (OUT_DIR / "conversion_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for path in (Q4_PATH, MMPROJ_PATH):
            print(f"{path.name}: {path.stat().st_size / 1024**3:.2f} GiB · {metadata['files'][path.name]['sha256']}")
        """
    ),
    code(
        r"""
        # 5. 不公開 sample 內容的 ChartQA runtime smoke verification
        #
        # 題目與標籤只在執行期從 pinned dataset revision 取得。
        # query hash 只是對齊識別碼，不是匿名化或隱私保證。
        import hashlib, json, re, subprocess
        from datasets import load_dataset

        QUERY_SHA256 = "b52c3a183c609a0b53978011daafbf8eb4fd886f4a3879ee76ae6325e0c2077d"

        def _sha256(text):
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        row = load_dataset(DATASET_ID, split="test", revision=DATASET_REVISION)[41]
        assert _sha256(row["query"]) == QUERY_SHA256, "ChartQA row 41 query 與 pinned 雜湊不符"
        raw_image = Path("/content/chartqa_raw_case_41.png")
        row["image"].convert("RGB").save(raw_image)

        help_text = subprocess.run([str(MTMD_BIN), "--help"], capture_output=True, text=True, check=True).stdout
        for flag in ("--mmproj", "--image", "--threads", "--threads-batch"):
            assert flag in help_text, f"pinned llama-mtmd-cli 缺少必要參數: {flag}"

        prompt = row["query"] + "\nAnswer the question using a single word or phrase."
        proc = subprocess.run([
            str(MTMD_BIN),
            "--model", str(Q4_PATH),
            "--mmproj", str(MMPROJ_PATH),
            "--image", str(raw_image),
            "--prompt", prompt,
            "--temp", "0",
            "--top-k", "1",
            "--top-p", "1",
            "--seed", "3407",
            "--n-predict", "64",
            "--ctx-size", "4096",
            "--threads", str(BUILD_JOBS),
            "--threads-batch", str(BUILD_JOBS),
        ], capture_output=True, text=True, timeout=15 * 60)
        print(proc.stdout[-3000:])
        if proc.returncode != 0:
            print(proc.stderr[-5000:])
        assert proc.returncode == 0, f"GGUF inference 失敗（exit={proc.returncode}）"
        gold = row["label"][0]
        passed = re.search(rf"(?<!\d){re.escape(gold)}(?!\d)", proc.stdout) is not None
        assert passed, f"GGUF smoke 答案未包含 gold：\n{proc.stdout[-2000:]}"

        metadata = json.loads((OUT_DIR / "conversion_metadata.json").read_text(encoding="utf-8"))
        metadata["smoke_verification"] = {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "split": "test",
            "row_index": 41,
            "query_sha256": QUERY_SHA256,
            "passed": True,
            "backend": "CPU",
            "threads": BUILD_JOBS,
            "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
        }
        (OUT_DIR / "conversion_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        card = f'''---
        license: apache-2.0
        base_model: {MERGED_REPO}
        library_name: llama.cpp
        pipeline_tag: image-text-to-text
        tags:
        - qwen3-vl
        - vision-language
        - chartqa
        - gguf
        - llama.cpp
        - q4-k-m
        datasets:
        - {DATASET_ID}
        ---

        # Qwen3-VL-8B ChartQA — GGUF Q4_K_M

        Persistent CPU-demo artifact converted from the pinned fine-tuned merged checkpoint
        [`{MERGED_REPO}@{MERGED_REVISION}`](https://huggingface.co/{MERGED_REPO}/tree/{MERGED_REVISION}).

        ## Files

        - `{Q4_PATH.name}` — fine-tuned language model, Q4_K_M
        - `{MMPROJ_PATH.name}` — fine-tuned vision encoder/projector, Q8_0

        Conversion used llama.cpp commit `{LLAMA_CPP_COMMIT}`. `conversion_metadata.json` records byte sizes,
        SHA-256 digests and the content-free outcome of the runtime ChartQA smoke verification.

        ## Verification scope

        One pinned ChartQA test sample passed an exact-match check against the label fetched at runtime.
        The sample, label, and model output are not published. This is a smoke verification, not a complete
        GGUF quality evaluation.

        The formal 2,500-question quality result belongs to the separately evaluated
        [AWQ/vLLM artifact](https://huggingface.co/steven0226/qwen3vl-8b-chartqa-awq): 85.52%,
        -0.72 pp versus merged 16-bit, passing the predefined 2 pp gate. Do not treat that number as a GGUF score.

        ## llama.cpp

        Use a recent llama.cpp build with Qwen3-VL support:

        ```bash
        llama-server \\
          --model {Q4_PATH.name} \\
          --mmproj {MMPROJ_PATH.name} \\
          --ctx-size 4096 --jinja
        ```

        ## 中文摘要

        這是 ChartQA fine-tuned Qwen3-VL-8B 的持久展示用 GGUF。語言模型採 Q4_K_M，vision encoder／projector
        採 Q8_0。已用不含答案標註的 raw ChartQA 圖表完成單題 smoke verification；正式品質數字仍以 AWQ/vLLM
        的完整 2,500 題評估為準，不跨 inference stack 混用。
        '''
        card = "\n".join(line[8:] if line.startswith("        ") else line for line in card.splitlines()).strip() + "\n"
        (OUT_DIR / "README.md").write_text(card, encoding="utf-8")
        print("GGUF CPU smoke verification PASS")
        """
    ),
    code(
        r"""
        # 6. 建立／更新 GGUF model repo，並驗證正式 revision
        from huggingface_hub import HfApi

        api.create_repo(GGUF_REPO, repo_type="model", private=False, exist_ok=True)
        commit = api.upload_folder(
            repo_id=GGUF_REPO,
            repo_type="model",
            folder_path=OUT_DIR,
            commit_message=f"Publish fine-tuned Q4_K_M GGUF from {MERGED_REVISION[:12]}",
        )
        final = retry("GGUF model_info", lambda: api.model_info(GGUF_REPO, files_metadata=True))
        sizes = {item.rfilename: int(item.size or 0) for item in final.siblings}
        for required in (Q4_PATH.name, MMPROJ_PATH.name, "conversion_metadata.json", "README.md"):
            assert required in sizes, f"Hub 缺少 {required}"
        assert sizes[Q4_PATH.name] == Q4_PATH.stat().st_size
        assert sizes[MMPROJ_PATH.name] == MMPROJ_PATH.stat().st_size
        print("GGUF_REPO:", f"https://huggingface.co/{GGUF_REPO}")
        print("GGUF_REVISION:", final.sha)
        print("MODEL_FILE:", Q4_PATH.name, sizes[Q4_PATH.name])
        print("MMPROJ_FILE:", MMPROJ_PATH.name, sizes[MMPROJ_PATH.name])
        print("PASS: fine-tuned GGUF 已發布；請記錄以上四行作為發布憑證。")
        """
    ),
]

output = Path(__file__).with_name("convert_gguf_colab_cpu_fixed.ipynb")
nbf.write(nb, output)
print(output)
