"""Maintenance source for the reviewed Stage 4 Colab notebook.

Run from the repository root when the notebook must be regenerated:
    uv run --with nbformat==5.5.0 python notebooks/_build_benchmark_notebook.py
"""

from __future__ import annotations

import textwrap

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
        # Qwen3-VL ChartQA — vLLM serving benchmark（Colab A100）

        這份 notebook **只執行 Phase 4 Stage 4**：在同一張 A100 上公平比較 merged 16-bit 與正式 AWQ W4A16-g32 的 serving 效能。

        使用方式：

        1. 執行階段選 **A100 GPU**，並確認 Colab Secret `HF_TOKEN` 已允許這份筆記本存取。
        2. 直接按「**全部執行**」，不需要修改任何程式碼。
        3. 預估約 30–60 分鐘；大量 log 會寫入 `/content/bench_vllm/`，畫面每分鐘只顯示一次心跳，避免瀏覽器卡頓。
        4. 每個 concurrency 完成後都會立即同步到 HF Hub；同一個 Colab runtime 中斷時，重新「全部執行」會驗證後續跑。若 Colab 配發了不同實體 GPU，程式會為了公平性自動重跑，不會混用舊數字。

        方法：使用 [vLLM 0.25.1 官方 `vllm bench serve`](https://docs.vllm.ai/en/v0.25.1/cli/bench/serve/)，每個模型在獨立 server process 執行。並發數為 1/4/8/16，每個 level 量測同一組 64 個真實 ChartQA 請求，固定輸出 64 tokens（`ignore_eos`）以穩定量測 TTFT、TPOT、E2E 與吞吐。各 level 保留相同尺寸／prompt，只以右下角 12×12 像素產生不同 image hash；每張正式圖另有獨立 warmup 版本，避免 multimodal encoder cache 污染數字。
        """
    ),
    code(
        r"""
        # 1. 硬體與磁碟檢查（vLLM 安裝前不要 import torch）
        import shutil, subprocess

        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        gpu_memory_total_mb = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        gpu_uuid = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        gpu_driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        disk_free_gb = shutil.disk_usage("/content").free / 1024**3
        print("GPU:", gpu)
        print("GPU UUID:", gpu_uuid)
        print("NVIDIA driver:", gpu_driver)
        print(f"Disk free: {disk_free_gb:.1f} GB")
        assert "A100" in gpu, "這份 benchmark 需要 A100 40GB；請更換執行階段後重新全部執行。"
        assert 39000 <= gpu_memory_total_mb <= 42000, (
            f"這份 benchmark 固定使用 A100 40GB，但目前 memory.total={gpu_memory_total_mb} MiB。"
        )
        assert "\n" not in gpu_uuid and "\n" not in gpu_driver, "這份 notebook 只支援單張 GPU。"
        assert disk_free_gb >= 45, "可用磁碟需至少 45GB；請先刪除其他 runtime 檔案。"
        """
    ),
    code(
        r"""
        %%capture
        # 2. 安裝已在本專案正式評估驗證過的 CUDA 12.9 套件組合
        import os, subprocess, sys

        os.environ.update({
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        })
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)
        subprocess.run([
            "uv", "pip", "install", "--system", "--no-cache",
            "vllm[bench]==0.25.1+cu129",
            "torch==2.11.0+cu129", "torchvision==0.26.0+cu129", "torchaudio==2.11.0+cu129",
            "transformers==5.10.1", "datasets==5.0.0",
            "pillow==11.3.0", "pandas==2.3.1", "matplotlib==3.10.3",
            "tabulate==0.9.0", "httpx==0.28.1",
            "--extra-index-url", "https://wheels.vllm.ai/0.25.1/cu129",
            "--extra-index-url", "https://download.pytorch.org/whl/cu129",
            "--index-strategy", "unsafe-best-match",
        ], check=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "hf_xet"], check=False)
        """
    ),
    code(
        r"""
        # 3. 套件 preflight、HF 登入、正式品質門檻與模型身分驗證
        import hashlib, importlib.metadata, json, os, shutil, subprocess, sys, time
        from pathlib import Path

        from google.colab import userdata
        from huggingface_hub import HfApi, hf_hub_download, login, whoami

        expected_versions = {
            "vllm": "0.25.1+cu129", "torch": "2.11.0+cu129",
            "torchvision": "0.26.0+cu129", "torchaudio": "2.11.0+cu129",
            "transformers": "5.10.1", "datasets": "5.0.0",
            "pillow": "11.3.0", "pandas": "2.3.1", "matplotlib": "3.10.3",
            "tabulate": "0.9.0", "httpx": "0.28.1",
        }
        resolved_versions = {name: importlib.metadata.version(name) for name in expected_versions}
        assert resolved_versions == expected_versions, resolved_versions
        recorded_versions = {
            **resolved_versions,
            "numpy": importlib.metadata.version("numpy"),
        }

        probe_code = (
            "import torch, torchvision; from vllm import LLM, SamplingParams; "
            "assert torch.version.cuda == '12.9', torch.version.cuda; "
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            "print('vLLM CUDA import OK:', torch.__version__, torch.version.cuda)"
        )
        probe = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
        print(probe.stdout.strip())
        if probe.returncode != 0:
            print(probe.stderr)
        assert probe.returncode == 0, "vLLM CUDA 12.9 import 失敗；請勿繼續下載模型。"

        VLLM_BIN = shutil.which("vllm")
        assert VLLM_BIN, "找不到 vllm CLI"
        serve_help = subprocess.run([VLLM_BIN, "serve", "--help=all"], capture_output=True, text=True, check=True).stdout
        bench_help = subprocess.run([VLLM_BIN, "bench", "serve", "--help=all"], capture_output=True, text=True, check=True).stdout
        for flag in ["--limit-mm-per-prompt", "--mm-processor-cache-gb", "--async-scheduling",
                     "--stream-interval", "--no-enable-prefix-caching", "--disable-uvicorn-access-log",
                     "--jit-monitor-verbose"]:
            assert flag in serve_help, f"vLLM serve 缺少必要參數: {flag}"
        for flag in ["--dataset-name", "--custom-ensure-client-side-data", "--enable-multimodal-chat",
                     "--custom-output-len", "--max-concurrency", "--ignore-eos", "--save-detailed"]:
            assert flag in bench_help, f"vLLM bench serve 缺少必要參數: {flag}"

        HF_TOKEN = userdata.get("HF_TOKEN")
        assert HF_TOKEN, "找不到 Colab Secret: HF_TOKEN"
        os.environ["HF_TOKEN"] = HF_TOKEN
        login(token=HF_TOKEN, add_to_git_credential=False)
        HF_USER = whoami()["name"]
        api = HfApi(token=HF_TOKEN)
        MERGED_REPO = f"{HF_USER}/qwen3vl-8b-chartqa-merged-16bit"
        AWQ_REPO = f"{HF_USER}/qwen3vl-8b-chartqa-awq"
        for repo in (MERGED_REPO, AWQ_REPO):
            assert api.repo_exists(repo), f"找不到模型 repo: {repo}"

        gate_path = hf_hub_download(AWQ_REPO, "eval_quant/results.json", token=HF_TOKEN)
        EXPECTED_QUALITY_GATE_SHA256 = "decd099d7c83065a53e36450c04d06fd96446e53e40dec9ba99e5a7cee4b1f22"
        QUALITY_GATE_SHA256 = hashlib.sha256(Path(gate_path).read_bytes()).hexdigest()
        assert QUALITY_GATE_SHA256 == EXPECTED_QUALITY_GATE_SHA256, (
            "HF 上的正式 eval_quant/results.json 已變更；禁止用未審核的品質門檻執行 benchmark。",
            QUALITY_GATE_SHA256,
        )
        gate = json.load(open(gate_path, encoding="utf-8"))
        overall = next(row for row in gate["table"] if row["test split"] == "overall")
        merged_acc = float(overall["merged-16bit"])
        awq_acc = float(overall["awq-w4a16-g32"])
        drop_pp = (merged_acc - awq_acc) * 100
        assert gate["quality_gate_passed"] is True, "正式 AWQ 品質門檻尚未通過，禁止 benchmark。"
        assert gate["eval_n_per_split"] == 1250 and int(overall["n"]) == 2500, gate
        assert drop_pp <= float(gate["quality_gate_max_drop_pp"]) + 1e-9, gate

        EXPECTED = {
            "merged16": {
                "repo": MERGED_REPO,
                "revision": "519060ef43df3261e0512e5ae4c82a4d4e675f32",
                "fingerprint": "62b1168902151c870f8b361d894a7af66264c68cc4e90219a0c8a828f85eb4a6",
            },
            "awq": {
                "repo": AWQ_REPO,
                "revision": "43b71926a1d645133560347787539729bcd3de6b",
                "fingerprint": "22c674970d90a22b2e87b6b3a630df2b5e7e2f5cbcb0a3dff65e404f548d9f31",
            },
        }

        def model_identity(repo, revision):
            info = api.model_info(repo, revision=revision, files_metadata=True)
            parts, weight_bytes = [], 0
            for item in info.siblings:
                name = item.rfilename
                if "/" in name or not name.endswith((".json", ".jinja", ".model", ".safetensors", ".txt")):
                    continue
                if name == "quantization_metadata.json":
                    continue
                digest = getattr(item.lfs, "sha256", None) if item.lfs else item.blob_id
                parts.append(f"{name}:{digest}")
                if name.endswith(".safetensors"):
                    weight_bytes += int(getattr(item, "size", 0) or 0)
            assert any(".safetensors:" in part for part in parts), f"找不到模型權重: {repo}@{revision}"
            fingerprint = hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()
            return info.sha, fingerprint, weight_bytes

        MODEL_SPECS = {}
        for tag, spec in EXPECTED.items():
            sha, fingerprint, weight_bytes = model_identity(spec["repo"], spec["revision"])
            assert sha == spec["revision"], (tag, sha)
            assert fingerprint == spec["fingerprint"], (tag, fingerprint)
            runtime = gate["runtime"]["merged" if tag == "merged16" else "awq"]
            assert runtime["model_revision"] == spec["revision"], runtime
            assert runtime["model_fingerprint"] == spec["fingerprint"], runtime
            MODEL_SPECS[tag] = {**spec, "weight_bytes": weight_bytes}

        DATASET_ID = "HuggingFaceM4/ChartQA"
        DATASET_REVISION = gate["runtime"]["merged"]["dataset_revision"]
        assert DATASET_REVISION == gate["runtime"]["awq"]["dataset_revision"]
        assert DATASET_REVISION == "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
        print("versions:", recorded_versions)
        print(f"品質門檻 PASS: merged={merged_acc:.2%}, AWQ={awq_acc:.2%}, 下降 {drop_pp:.2f}pp")
        print("HF user:", HF_USER)
        for tag, spec in MODEL_SPECS.items():
            print(f"{tag}: {spec['revision'][:12]} | weights {spec['weight_bytes']/1e9:.2f} GB")
        """
    ),
    code(
        r"""
        # 4. 建立固定、可重現且跨 concurrency 不重複的 ChartQA 工作負載
        import hashlib, io, json
        from pathlib import Path

        from datasets import Dataset, disable_progress_bars
        from huggingface_hub import hf_hub_download
        from PIL import Image, ImageDraw, ImageOps

        disable_progress_bars()
        SEED = 3407
        CONCURRENCIES = [1, 4, 8, 16]
        N_PER_LEVEL = 64
        ANSWER_INSTRUCTION = "Answer the question using a single word or phrase."
        WORK_DIR = Path("/content/bench_vllm")
        IMAGE_DIR = WORK_DIR / "images"
        WORKLOAD_DIR = WORK_DIR / "workloads"
        CHECKPOINT_DIR = WORK_DIR / "checkpoints"
        LOG_DIR = WORK_DIR / "logs"
        for directory in (IMAGE_DIR, WORKLOAD_DIR, CHECKPOINT_DIR, LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        test_files = sorted(
            f for f in api.list_repo_files(DATASET_ID, repo_type="dataset", revision=DATASET_REVISION)
            if f.startswith("data/test-") and f.endswith(".parquet")
        )
        assert test_files, "ChartQA test parquet not found"
        parquet_paths = [
            hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=DATASET_REVISION)
            for name in test_files
        ]
        ds = Dataset.from_parquet(parquet_paths)
        ds = ds.add_column("_source_index", list(range(len(ds))))
        human = ds.filter(lambda ex: ex["human_or_machine"] == 0).shuffle(seed=SEED)
        augmented = ds.filter(lambda ex: ex["human_or_machine"] == 1).shuffle(seed=SEED)

        def standardized(image, max_side=1024):
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side))
            return image

        def jpeg_bytes(image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()

        def save_jpeg(image, path):
            Path(path).write_bytes(jpeg_bytes(image))

        def file_sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def canonical_sha(value):
            blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            return hashlib.sha256(blob).hexdigest()

        # ChartQA 的同一張圖可能對應多個問題；先依標準化後的 JPEG hash 去重。
        # 四個 concurrency 共用同一組 64 題，才能把差異歸因於 concurrency；
        # 各 level 再產生同尺寸、極小像素差異的版本，避開 encoder cache hash。
        def select_unique_source_indices(dataset, needed, seen_hashes):
            selected = []
            for index in range(len(dataset)):
                row = dataset[index]
                digest = hashlib.sha256(jpeg_bytes(standardized(row["image"]))).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                selected.append({"source_index": int(row["_source_index"]), "image_sha256": digest})
                if len(selected) == needed:
                    return selected
            raise RuntimeError(f"唯一圖表不足：需要 {needed}，只找到 {len(selected)}")

        selected_hashes = set()
        selected_human = select_unique_source_indices(human, 33, selected_hashes)
        selected_augmented = select_unique_source_indices(augmented, 32, selected_hashes)
        assert len(selected_hashes) == 65

        probe_row = ds[selected_human[0]["source_index"]]
        probe_image = standardized(probe_row["image"])
        PROBE_PATH = IMAGE_DIR / "probe.jpg"
        save_jpeg(ImageOps.flip(probe_image), PROBE_PATH)

        def marked_variant(image, code):
            image = image.copy()
            width, height = image.size
            block = min(12, width, height)
            color = ((37 * code + 17) % 256, (97 * code + 53) % 256, (193 * code + 91) % 256)
            ImageDraw.Draw(image).rectangle(
                [width - block, height - block, width - 1, height - 1], fill=color
            )
            return image

        def decoded_pixel_sha(path):
            with Image.open(path) as decoded:
                decoded = decoded.convert("RGB")
                payload = f"{decoded.width}x{decoded.height}:".encode() + decoded.tobytes()
            return hashlib.sha256(payload).hexdigest()

        base_rows = []
        for offset in range(32):
            human_item = selected_human[1 + offset]
            augmented_item = selected_augmented[offset]
            base_rows.append(("human", ds[human_item["source_index"]], human_item["image_sha256"]))
            base_rows.append(("augmented", ds[augmented_item["source_index"]], augmented_item["image_sha256"]))

        LEVELS = {}
        manifest_levels = []
        encoded_hashes, decoded_hashes = [], []
        for level_index, concurrency in enumerate(CONCURRENCIES):
            measured_lines, warm_lines, samples = [], [], []
            for request_index, (split, row, selected_image_sha) in enumerate(base_rows):
                sample_id = f"c{concurrency:02d}-r{request_index:03d}-{split[0]}"
                base_image = standardized(row["image"])
                assert hashlib.sha256(jpeg_bytes(base_image)).hexdigest() == selected_image_sha
                measured_path = IMAGE_DIR / f"{sample_id}.jpg"
                warm_path = IMAGE_DIR / f"{sample_id}-warm.jpg"
                measured_image = marked_variant(base_image, 10 + level_index)
                warm_image = marked_variant(base_image, 100 + level_index)
                width, height = base_image.size
                save_jpeg(measured_image, measured_path)
                save_jpeg(warm_image, warm_path)

                measured_sha, warm_sha = file_sha(measured_path), file_sha(warm_path)
                assert measured_sha != warm_sha
                measured_pixel_sha, warm_pixel_sha = decoded_pixel_sha(measured_path), decoded_pixel_sha(warm_path)
                assert measured_pixel_sha != warm_pixel_sha
                encoded_hashes.extend([measured_sha, warm_sha])
                decoded_hashes.extend([measured_pixel_sha, warm_pixel_sha])
                prompt = f"{row['query']}\n{ANSWER_INSTRUCTION}"
                measured_lines.append({"content": [
                    {"type": "image", "image": str(measured_path)},
                    {"type": "text", "text": prompt},
                ]})
                warm_lines.append({"content": [
                    {"type": "image", "image": str(warm_path)},
                    {"type": "text", "text": prompt},
                ]})
                samples.append({
                    "sample_id": sample_id, "split": split,
                    "source_index": int(row["_source_index"]),
                    "query_sha256": hashlib.sha256(str(row["query"]).encode()).hexdigest(),
                    "base_image_sha256": selected_image_sha,
                    "image_sha256": measured_sha, "warm_image_sha256": warm_sha,
                    "decoded_pixel_sha256": measured_pixel_sha,
                    "warm_decoded_pixel_sha256": warm_pixel_sha,
                    "width": width, "height": height,
                })

            measured_path = WORKLOAD_DIR / f"measure_c{concurrency:02d}.jsonl"
            warm_path = WORKLOAD_DIR / f"warm_c{concurrency:02d}.jsonl"
            for path, lines in ((measured_path, measured_lines), (warm_path, warm_lines)):
                with open(path, "w", encoding="utf-8") as handle:
                    for line in lines:
                        handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            level_record = {
                "concurrency": concurrency, "n_measured": len(measured_lines),
                "n_warmup": len(warm_lines), "measured_file": measured_path.name,
                "warmup_file": warm_path.name,
                "measured_jsonl_sha256": file_sha(measured_path),
                "warmup_jsonl_sha256": file_sha(warm_path),
                "samples": samples,
            }
            level_sha = canonical_sha(level_record)
            level_record["sha256"] = level_sha
            manifest_levels.append(level_record)
            LEVELS[concurrency] = {
                "measure": measured_path, "warm": warm_path,
                "sample_ids": [sample["sample_id"] for sample in samples],
                "sha256": level_sha,
            }

        assert len(encoded_hashes) == len(set(encoded_hashes)), "工作負載出現重複 encoded image hash"
        assert len(decoded_hashes) == len(set(decoded_hashes)), "工作負載出現重複 decoded pixel hash"
        assert file_sha(PROBE_PATH) not in set(encoded_hashes), "probe 與正式影像 encoded hash 重複"
        assert decoded_pixel_sha(PROBE_PATH) not in set(decoded_hashes), "probe 與正式影像 pixel hash 重複"
        cohorts = [[(s["source_index"], s["query_sha256"]) for s in level["samples"]]
                   for level in manifest_levels]
        assert all(cohort == cohorts[0] for cohort in cohorts[1:]), "各 concurrency cohort 不一致"
        workload_manifest = {
            "schema_version": 2, "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "seed": SEED, "image_transform": "RGB; thumbnail<=1024; JPEG q90",
            "selection": "shuffle each split with seed 3407; first unique standardized-image SHA; human[0] reserved for probe; same 32 human + 32 augmented cohort at every level",
            "prompt": {"content_order": ["image", "text"], "answer_instruction": ANSWER_INSTRUCTION},
            "cache_control": "same source cohort; per-level measured/warm variants preserve dimensions and differ by a 12x12 corner marker; encoded and decoded-pixel hashes all unique",
            "levels": manifest_levels,
        }
        WORKLOAD_SHA256 = canonical_sha(workload_manifest)
        workload_manifest["workload_sha256"] = WORKLOAD_SHA256
        MANIFEST_PATH = WORK_DIR / "workload_manifest.json"
        json.dump(workload_manifest, open(MANIFEST_PATH, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"工作負載完成: 同一組 64 題 × 4 levels；每個 level 有獨立 measured/warm 圖片 hash")
        print("workload sha256:", WORKLOAD_SHA256)
        """
    ),
    code(
        r"""
        # 5. vLLM server、官方 benchmark、續跑與嚴格驗證工具
        import base64, datetime as dt, json, math, os, shutil, signal, socket, subprocess, time, uuid
        import urllib.error, urllib.request
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
        from pathlib import Path

        from huggingface_hub import CommitOperationAdd, hf_hub_download, snapshot_download

        SERVED_MODEL_NAME = "chartqa-benchmark"
        HOST, PORT = "127.0.0.1", 8000
        DRIVER_REVISION = "stage4-official-v4"
        ATTEMPT_ID = uuid.uuid4().hex[:12]
        OUTPUT_TOKENS = 64
        WARMUP_OUTPUT_TOKENS = 64
        SERVER_START_TIMEOUT = 20 * 60
        BENCH_TIMEOUT = 30 * 60
        SERVER_CONFIG = {
            "dtype": "auto", "max_model_len": 4096, "max_num_seqs": 16,
            "gpu_memory_utilization": 0.88, "limit_mm_per_prompt": {"image": 1, "video": 0},
            "mm_processor_cache_gb": 0, "prefix_caching": False,
            "async_scheduling": True, "stream_interval": 1,
            "generation_config": "vllm", "enforce_eager": False,
            "jit_monitor_verbose": True, "seed": SEED,
        }
        recipe = {
            "schema_version": 3, "driver_revision": DRIVER_REVISION,
            "engine": "vllm bench serve 0.25.1+cu129 / openai-chat",
            "gpu": gpu, "gpu_uuid": gpu_uuid, "nvidia_driver": gpu_driver,
            "packages": resolved_versions, "dataset_revision": DATASET_REVISION,
            "quality_gate_sha256": QUALITY_GATE_SHA256,
            "workload_sha256": WORKLOAD_SHA256,
            "models": {tag: {key: spec[key] for key in ("repo", "revision", "fingerprint")}
                       for tag, spec in MODEL_SPECS.items()},
            "server": SERVER_CONFIG, "concurrencies": CONCURRENCIES,
            "measured_requests_per_level": N_PER_LEVEL,
            "warmup_requests_per_level": N_PER_LEVEL,
            "warmup_output_tokens": WARMUP_OUTPUT_TOKENS,
            "measured_output_tokens": OUTPUT_TOKENS,
            "sampling": {"temperature": 0, "ignore_eos": True, "request_rate": "inf"},
            "metric_percentiles": [50, 95], "server_order": ["merged16", "awq"],
        }
        RECIPE_SHA256 = hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        RUN_ID = f"v2-{RECIPE_SHA256[:12]}"
        print("benchmark run_id:", RUN_ID, "| attempt:", ATTEMPT_ID)

        HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def tail(path, lines=60):
            path = Path(path)
            text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            return "\n".join(text.splitlines()[-lines:])

        def upload_with_retry(path, remote_path, attempts=6):
            for attempt in range(1, attempts + 1):
                try:
                    api.upload_file(path_or_fileobj=str(path), path_in_repo=remote_path,
                                    repo_id=AWQ_REPO, token=HF_TOKEN)
                    return
                except Exception as exc:
                    if attempt == attempts:
                        raise
                    wait = min(15 * attempt, 60)
                    print(f"upload retry {attempt}/{attempts}: {type(exc).__name__}; {wait}s")
                    time.sleep(wait)

        def commit_files_with_retry(files, message, attempts=6):
            operations = [
                CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                for local, remote in files
            ]
            for attempt in range(1, attempts + 1):
                try:
                    api.create_commit(
                        repo_id=AWQ_REPO, repo_type="model", operations=operations,
                        commit_message=message, token=HF_TOKEN,
                    )
                    return
                except Exception as exc:
                    if "no files have changed" in str(exc).lower():
                        return
                    if attempt == attempts:
                        raise
                    wait = min(15 * attempt, 60)
                    print(f"atomic commit retry {attempt}/{attempts}: {type(exc).__name__}; {wait}s")
                    time.sleep(wait)

        def descendants(root_pid):
            result = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True)
            children = {}
            for line in result.stdout.splitlines():
                try:
                    pid, ppid = map(int, line.split())
                except ValueError:
                    continue
                children.setdefault(ppid, []).append(pid)
            found, stack = set(), [root_pid]
            while stack:
                parent = stack.pop()
                for child in children.get(parent, []):
                    if child not in found:
                        found.add(child)
                        stack.append(child)
            return found

        def terminate_group(proc):
            if proc is None:
                return
            known = {proc.pid} | descendants(proc.pid)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            known |= descendants(proc.pid)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            for pid in known:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        def run_logged(cmd, log_path, label, timeout=BENCH_TIMEOUT, env=None):
            log_path = Path(log_path)
            started = time.time()
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                        env=env or os.environ.copy(), start_new_session=True)
            try:
                while True:
                    try:
                        returncode = proc.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        elapsed = time.time() - started
                        size_mb = log_path.stat().st_size / 1024**2 if log_path.exists() else 0
                        print(f"[{label}] still running: {elapsed/60:.0f} min | log {size_mb:.1f} MB")
                        if elapsed > timeout:
                            raise TimeoutError(f"{label} 超過 {timeout/60:.0f} 分鐘")
            except BaseException:
                terminate_group(proc)
                print(tail(log_path, 80))
                raise
            if returncode != 0:
                print(tail(log_path, 80))
                raise RuntimeError(f"{label} 失敗（exit={returncode}）")

        def get_json(url, timeout=10):
            with HTTP.open(url, timeout=timeout) as response:
                return json.loads(response.read().decode())

        def health_reachable():
            try:
                with HTTP.open(f"http://{HOST}:{PORT}/health", timeout=2) as response:
                    return response.status == 200
            except Exception:
                return False

        def port_bindable():
            probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe_socket.bind((HOST, PORT))
                return True
            except OSError:
                return False
            finally:
                probe_socket.close()

        def gpu_compute_processes():
            return subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True,
            ).stdout.strip()

        def assert_clean_start():
            compute = gpu_compute_processes()
            assert not health_reachable(), "Port 8000 已有舊 vLLM server；請重新啟動 Colab runtime。"
            assert port_bindable(), "Port 8000 無法綁定；請重新啟動 Colab runtime。"
            assert not compute, f"啟動前仍有 GPU process；請重新啟動 Colab runtime：\n{compute}"

        def wait_for_server(proc, log_path):
            started, last_notice = time.time(), -1
            while time.time() - started < SERVER_START_TIMEOUT:
                if proc.poll() is not None:
                    print(tail(log_path, 100))
                    raise RuntimeError(f"vLLM server 提前退出（exit={proc.returncode}）")
                try:
                    with HTTP.open(f"http://{HOST}:{PORT}/health", timeout=5) as response:
                        if response.status == 200:
                            models = get_json(f"http://{HOST}:{PORT}/v1/models")
                            ids = [item["id"] for item in models.get("data", [])]
                            assert SERVED_MODEL_NAME in ids, ids
                            return
                except Exception:
                    pass
                elapsed_min = int((time.time() - started) // 60)
                if elapsed_min > last_notice:
                    print(f"[server] 載入中: {elapsed_min} min")
                    last_notice = elapsed_min
                time.sleep(5)
            print(tail(log_path, 100))
            raise TimeoutError("vLLM server 20 分鐘內未就緒")

        def data_uri(path):
            encoded = base64.b64encode(Path(path).read_bytes()).decode()
            return "data:image/jpeg;base64," + encoded

        def functional_probe():
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri(PROBE_PATH)}},
                    {"type": "text", "text": f"Describe this chart briefly.\n{ANSWER_INSTRUCTION}"},
                ]}],
                "temperature": 0, "max_tokens": 4,
            }
            request = urllib.request.Request(
                f"http://{HOST}:{PORT}/v1/chat/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
            )
            with HTTP.open(request, timeout=180) as response:
                body = json.loads(response.read().decode())
            assert body.get("choices"), body

        def gpu_memory_used_mb():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            return int(result.stdout.strip().splitlines()[0])

        def snapshot_with_heartbeat(repo, revision, tag):
            started = time.time()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(snapshot_download, repo, revision=revision, token=HF_TOKEN)
                while True:
                    try:
                        return future.result(timeout=60)
                    except FutureTimeout:
                        print(f"[{tag}] model snapshot still downloading: {(time.time()-started)/60:.0f} min")

        def start_server(tag):
            spec = MODEL_SPECS[tag]
            assert_clean_start()
            print(f"[{tag}] 取得 pinned model snapshot（已快取時不會重抓權重）")
            model_path = snapshot_with_heartbeat(spec["repo"], spec["revision"], tag)
            log_path = LOG_DIR / f"server_{tag}_{ATTEMPT_ID}.log"
            cmd = [
                VLLM_BIN, "serve", model_path,
                "--host", HOST, "--port", str(PORT),
                "--served-model-name", SERVED_MODEL_NAME,
                "--dtype", "auto", "--max-model-len", "4096", "--max-num-seqs", "16",
                "--gpu-memory-utilization", "0.88",
                "--limit-mm-per-prompt", '{"image":1,"video":0}',
                "--mm-processor-cache-gb", "0", "--no-enable-prefix-caching",
                "--async-scheduling", "--stream-interval", "1",
                "--generation-config", "vllm", "--seed", str(SEED),
                "--jit-monitor-verbose",
                "--disable-log-stats", "--disable-uvicorn-access-log",
                "--uvicorn-log-level", "warning",
            ]
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "VLLM_LOGGING_LEVEL": "INFO",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1"})
            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
            try:
                wait_for_server(proc, log_path)
                functional_probe()
                assert proc.poll() is None, "multimodal probe 後 server 已退出"
            except BaseException:
                terminate_group(proc)
                raise
            print(f"[{tag}] server ready；multimodal chat probe PASS")
            return proc, model_path, cmd, log_path

        def stop_server(proc):
            terminate_group(proc)
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    with HTTP.open(f"http://{HOST}:{PORT}/health", timeout=2):
                        time.sleep(2)
                except Exception:
                    break
            compute = ""
            cleanup_deadline = time.time() + 60
            while time.time() < cleanup_deadline:
                compute = gpu_compute_processes()
                if not compute:
                    break
                time.sleep(3)
            assert not compute, f"server 結束後仍有 GPU process：\n{compute}"
            assert not health_reachable(), "server 結束後 /health 仍可連線"
            assert port_bindable(), "server 結束後 Port 8000 仍被占用"

        def bench_command(model_path, dataset_path, concurrency, output_tokens,
                          result_path, detailed, request_id_prefix):
            cmd = [
                VLLM_BIN, "bench", "serve",
                "--backend", "openai-chat", "--host", HOST, "--port", str(PORT),
                "--endpoint", "/v1/chat/completions",
                "--model", model_path, "--served-model-name", SERVED_MODEL_NAME,
                "--dataset-name", "custom_image", "--dataset-path", str(dataset_path),
                "--custom-ensure-client-side-data", "--enable-multimodal-chat",
                "--no-oversample", "--disable-shuffle", "--num-prompts", str(N_PER_LEVEL),
                "--custom-output-len", str(output_tokens),
                "--request-rate", "inf", "--max-concurrency", str(concurrency),
                "--num-warmups", "0", "--ready-check-timeout-sec", "0",
                "--ignore-eos", "--temperature", "0", "--seed", str(SEED),
                "--request-id-prefix", request_id_prefix,
                "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
                "--disable-tqdm", "--save-result", "--result-filename", str(result_path),
            ]
            if detailed:
                cmd.append("--save-detailed")
            return cmd

        def expected_identity(tag, concurrency):
            spec = MODEL_SPECS[tag]
            return {
                "run_id": RUN_ID, "recipe_sha256": RECIPE_SHA256,
                "driver_revision": DRIVER_REVISION,
                "tag": tag, "model_repo": spec["repo"], "model_revision": spec["revision"],
                "model_fingerprint": spec["fingerprint"], "dataset_revision": DATASET_REVISION,
                "quality_gate_sha256": QUALITY_GATE_SHA256,
                "workload_sha256": WORKLOAD_SHA256, "level_sha256": LEVELS[concurrency]["sha256"],
                "concurrency": concurrency, "num_prompts": N_PER_LEVEL,
                "output_tokens": OUTPUT_TOKENS, "gpu": gpu, "gpu_uuid": gpu_uuid,
                "nvidia_driver": gpu_driver, "vllm": resolved_versions["vllm"],
            }

        def validate_official_result(result):
            assert int(result["completed"]) == N_PER_LEVEL, result.get("errors")
            assert int(result.get("failed", 0)) == 0, result.get("errors")
            assert int(result["total_output_tokens"]) == N_PER_LEVEL * OUTPUT_TOKENS
            assert len(result["output_lens"]) == N_PER_LEVEL
            assert all(int(length) == OUTPUT_TOKENS for length in result["output_lens"]), result["output_lens"]
            assert len(result["input_lens"]) == N_PER_LEVEL and min(result["input_lens"]) > 0
            assert max(result["input_lens"]) + OUTPUT_TOKENS <= SERVER_CONFIG["max_model_len"]
            assert not any(result.get("errors", [])), result.get("errors")
            for metric in ("ttft", "tpot", "e2el"):
                for percentile in (50, 95):
                    assert float(result[f"p{percentile}_{metric}_ms"]) > 0
            assert float(result["request_throughput"]) > 0
            assert float(result["output_throughput"]) > 0

        def checkpoint_path(tag, concurrency):
            return CHECKPOINT_DIR / f"{tag}_c{concurrency:02d}.json"

        def checkpoint_matches(path, tag, concurrency):
            try:
                wrapper = json.load(open(path, encoding="utf-8"))
                if wrapper.get("identity") != expected_identity(tag, concurrency):
                    return False
                session = wrapper.get("session", {})
                if not session.get("attempt_id") or not session.get("gpu_uuid"):
                    return False
                if wrapper.get("server", {}).get("jit_warning_during_measurement") is not False:
                    return False
                validate_official_result(wrapper["official_result"])
                return True
            except Exception:
                return False

        def load_checkpoint(tag, concurrency):
            local = checkpoint_path(tag, concurrency)
            if local.exists() and checkpoint_matches(local, tag, concurrency):
                return json.load(open(local, encoding="utf-8"))
            local.unlink(missing_ok=True)
            remote = f"bench/checkpoints/{RUN_ID}/{local.name}"
            try:
                cached = hf_hub_download(AWQ_REPO, remote, token=HF_TOKEN)
                shutil.copy2(cached, local)
                if checkpoint_matches(local, tag, concurrency):
                    print(f"[{tag} c={concurrency}] 使用 HF Hub 相符 checkpoint")
                    return json.load(open(local, encoding="utf-8"))
                local.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        def run_model(tag):
            cached = {c: load_checkpoint(tag, c) for c in CONCURRENCIES}
            if all(cached.values()):
                print(f"[{tag}] 四個 concurrency 均已完成，跳過 server 載入。")
                return cached

            proc, server_log = None, None
            try:
                proc, model_path, server_cmd, server_log = start_server(tag)
                server_memory_mb = gpu_memory_used_mb()
                for concurrency in CONCURRENCIES:
                    if cached[concurrency] is not None:
                        continue
                    level = LEVELS[concurrency]
                    print(f"[{tag} c={concurrency}] warmup 64 張同尺寸獨立影像（不計分）")
                    warm_raw = WORK_DIR / f"warm_raw_{tag}_c{concurrency:02d}.json"
                    warm_raw.unlink(missing_ok=True)
                    warm_log = LOG_DIR / f"warm_{tag}_c{concurrency:02d}.log"
                    run_logged(
                        bench_command(model_path, level["warm"], concurrency,
                                      WARMUP_OUTPUT_TOKENS, warm_raw, detailed=False,
                                      request_id_prefix=f"{tag}-warm-c{concurrency:02d}-{ATTEMPT_ID}-"),
                        warm_log, f"{tag} c={concurrency} warmup",
                    )
                    warm_result = json.load(open(warm_raw, encoding="utf-8"))
                    assert int(warm_result["completed"]) == N_PER_LEVEL
                    assert int(warm_result.get("failed", 0)) == 0
                    assert int(warm_result["total_output_tokens"]) == N_PER_LEVEL * WARMUP_OUTPUT_TOKENS
                    time.sleep(2)  # 讓 warmup 的 server log 完整 flush，再開始量測區段

                    measured_raw = WORK_DIR / f"official_{tag}_c{concurrency:02d}.json"
                    measured_raw.unlink(missing_ok=True)
                    measured_log = LOG_DIR / f"bench_{tag}_c{concurrency:02d}.log"
                    server_offset = server_log.stat().st_size if server_log.exists() else 0
                    print(f"[{tag} c={concurrency}] 正式量測 64 requests × 64 output tokens")
                    run_logged(
                        bench_command(model_path, level["measure"], concurrency,
                                      OUTPUT_TOKENS, measured_raw, detailed=True,
                                      request_id_prefix=f"{tag}-measure-c{concurrency:02d}-{ATTEMPT_ID}-"),
                        measured_log, f"{tag} c={concurrency} benchmark",
                    )
                    official = json.load(open(measured_raw, encoding="utf-8"))
                    validate_official_result(official)
                    time.sleep(2)  # 等待 server logger flush，避免漏掉量測尾端的 JIT warning
                    new_server_log = ""
                    if server_log.exists():
                        with open(server_log, "rb") as handle:
                            handle.seek(server_offset)
                            new_server_log = handle.read().decode("utf-8", errors="replace")
                    jit_warning = "JIT compilation during inference" in new_server_log
                    if jit_warning:
                        raise RuntimeError(
                            f"{tag} c={concurrency} 正式量測期間出現 JIT compilation；"
                            "此 level 不會保存。請重新『全部執行』，讓新 server 使用已產生的 kernel cache 重跑。"
                        )
                    wrapper = {
                        "identity": expected_identity(tag, concurrency),
                        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "session": {"attempt_id": ATTEMPT_ID, "gpu_uuid": gpu_uuid},
                        "server": {"command": server_cmd, "gpu_memory_used_mb": server_memory_mb,
                                   "log_file": server_log.name,
                                   "jit_warning_during_measurement": False},
                        "sample_ids": level["sample_ids"],
                        "official_result": official,
                    }
                    local = checkpoint_path(tag, concurrency)
                    json.dump(wrapper, open(local, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                    assert checkpoint_matches(local, tag, concurrency)
                    upload_with_retry(local, f"bench/checkpoints/{RUN_ID}/{local.name}")
                    cached[concurrency] = wrapper
                    print(
                        f"[{tag} c={concurrency}] req/s={official['request_throughput']:.2f} | "
                        f"tok/s={official['output_throughput']:.1f} | "
                        f"TTFT p95={official['p95_ttft_ms']:.1f} ms | checkpoint pushed"
                    )
                return cached
            finally:
                try:
                    if proc is not None:
                        stop_server(proc)
                        print(f"[{tag}] server 已完整結束，GPU 記憶體已釋放。")
                finally:
                    if server_log is not None and server_log.exists():
                        try:
                            upload_with_retry(
                                server_log,
                                f"bench/runs/{RUN_ID}/logs/{ATTEMPT_ID}/{server_log.name}",
                            )
                        except Exception as exc:
                            print(f"WARNING: server log 上傳失敗（不影響已驗證 checkpoint）: {type(exc).__name__}")
        """
    ),
    code(
        r"""
        # 6. 依序執行兩個模型；每個 level 皆可由 HF Hub 續跑
        MODEL_RESULTS = {}
        for tag in ("merged16", "awq"):
            print(f"\n========== {tag} ==========")
            MODEL_RESULTS[tag] = run_model(tag)
        print("\n兩個模型的 8 組正式 benchmark 全部完成。")
        """
    ),
    code(
        r"""
        # 7. 彙整、畫圖並同步正式 benchmark 產物到 HF Hub
        import json
        import matplotlib.pyplot as plt
        import pandas as pd

        rows = []
        for tag in ("merged16", "awq"):
            for concurrency in CONCURRENCIES:
                wrapper = MODEL_RESULTS[tag][concurrency]
                result = wrapper["official_result"]
                rows.append({
                    "model": "merged-16bit" if tag == "merged16" else "awq-w4a16-g32",
                    "tag": tag, "concurrency": concurrency, "n": int(result["completed"]),
                    "failed": int(result.get("failed", 0)),
                    "requests_per_s": float(result["request_throughput"]),
                    "output_tokens_per_s": float(result["output_throughput"]),
                    "ttft_p50_ms": float(result["p50_ttft_ms"]),
                    "ttft_p95_ms": float(result["p95_ttft_ms"]),
                    "tpot_p50_ms": float(result["p50_tpot_ms"]),
                    "tpot_p95_ms": float(result["p95_tpot_ms"]),
                    "e2e_p50_ms": float(result["p50_e2el_ms"]),
                    "e2e_p95_ms": float(result["p95_e2el_ms"]),
                    "mean_prompt_tokens": sum(result["input_lens"]) / len(result["input_lens"]),
                    "weight_gb": MODEL_SPECS[tag]["weight_bytes"] / 1e9,
                    "server_gpu_memory_mb": int(wrapper["server"]["gpu_memory_used_mb"]),
                    "attempt_id": wrapper["session"]["attempt_id"],
                    "gpu_uuid": wrapper["session"]["gpu_uuid"],
                    "jit_warning_during_measurement": bool(
                        wrapper["server"]["jit_warning_during_measurement"]
                    ),
                })

        comparisons = []
        for concurrency in CONCURRENCIES:
            merged = next(row for row in rows if row["tag"] == "merged16" and row["concurrency"] == concurrency)
            awq = next(row for row in rows if row["tag"] == "awq" and row["concurrency"] == concurrency)
            comparisons.append({
                "concurrency": concurrency,
                "awq_request_throughput_ratio": awq["requests_per_s"] / merged["requests_per_s"],
                "awq_output_throughput_ratio": awq["output_tokens_per_s"] / merged["output_tokens_per_s"],
                "awq_ttft_p95_ratio": awq["ttft_p95_ms"] / merged["ttft_p95_ms"],
                "awq_tpot_p95_ratio": awq["tpot_p95_ms"] / merged["tpot_p95_ms"],
            })

        table_columns = [
            "model", "concurrency", "n", "failed", "requests_per_s", "output_tokens_per_s",
            "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms",
            "e2e_p50_ms", "e2e_p95_ms",
        ]
        df = pd.DataFrame(rows)
        display_df = df[table_columns].copy()
        for column in table_columns[4:]:
            display_df[column] = display_df[column].round(2)
        print(display_df.to_markdown(index=False))

        jit_levels = [f"{row['tag']}:c{row['concurrency']}" for row in rows
                      if row["jit_warning_during_measurement"]]
        attempt_ids = sorted({row["attempt_id"] for row in rows})
        gpu_uuids = sorted({row["gpu_uuid"] for row in rows})
        input_token_workloads = [
            tuple(MODEL_RESULTS[tag][concurrency]["official_result"]["input_lens"])
            for tag in ("merged16", "awq") for concurrency in CONCURRENCIES
        ]
        same_input_token_workload = all(
            workload == input_token_workloads[0] for workload in input_token_workloads[1:]
        )
        same_cohort = all(
            [(sample["source_index"], sample["query_sha256"])
             for sample in level["samples"]]
            == [(sample["source_index"], sample["query_sha256"])
                for sample in workload_manifest["levels"][0]["samples"]]
            for level in workload_manifest["levels"][1:]
        )
        validity = {
            "all_requests_succeeded": all(row["failed"] == 0 and row["n"] == N_PER_LEVEL for row in rows),
            "fixed_output_tokens_verified": True,
            "same_source_cohort_across_levels": same_cohort,
            "same_input_token_workload_all_models_and_levels": same_input_token_workload,
            "encoded_and_decoded_image_hashes_unique": True,
            "processor_cache_disabled": True,
            "prefix_cache_disabled": True,
            "jit_warning_levels": jit_levels,
            "attempt_ids": attempt_ids, "gpu_uuids": gpu_uuids,
            "mixed_sessions": len(attempt_ids) > 1,
            "same_physical_gpu": gpu_uuids == [gpu_uuid],
        }
        validity["valid_for_reporting"] = (
            validity["all_requests_succeeded"]
            and validity["fixed_output_tokens_verified"]
            and validity["same_source_cohort_across_levels"]
            and validity["same_input_token_workload_all_models_and_levels"]
            and validity["encoded_and_decoded_image_hashes_unique"]
            and validity["processor_cache_disabled"]
            and validity["prefix_cache_disabled"]
            and validity["same_physical_gpu"]
            and not validity["jit_warning_levels"]
        )
        final_result = {
            "schema_version": 3, "run_id": RUN_ID,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "environment": {"gpu": gpu, "gpu_uuid": gpu_uuid, "nvidia_driver": gpu_driver,
                            "packages": recorded_versions},
            "quality_gate": {
                "n": 2500, "merged_accuracy": merged_acc, "awq_accuracy": awq_acc,
                "awq_change_pp": (awq_acc - merged_acc) * 100,
                "max_allowed_drop_pp": gate["quality_gate_max_drop_pp"], "passed": True,
                "source_sha256": QUALITY_GATE_SHA256,
            },
            "recipe_sha256": RECIPE_SHA256, "recipe": recipe,
            "workload_manifest": workload_manifest,
            "models": MODEL_SPECS, "rows": rows, "comparisons": comparisons,
            "validity": validity,
        }
        RESULT_PATH = WORK_DIR / "benchmark_results.json"
        TABLE_PATH = WORK_DIR / "benchmark_table.md"
        PLOT_PATH = WORK_DIR / "latency_throughput.png"
        json.dump(final_result, open(RESULT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

        markdown = "\n".join([
            "# Qwen3-VL ChartQA vLLM benchmark",
            "",
            f"- Run ID: `{RUN_ID}`",
            f"- GPU: {gpu}",
            f"- Quality gate: PASS ({merged_acc:.2%} → {awq_acc:.2%}, {(awq_acc-merged_acc)*100:+.2f}pp)",
            "- Workload: 64 real ChartQA requests per level; fixed 64 output tokens; ignore EOS",
            "- Fairness: same 64-source cohort at every concurrency; per-level variants preserve dimensions/prompts",
            "- Token workload: merged/AWQ input_lens are identical at all four concurrency levels",
            "- Cache control: encoded and decoded-pixel hashes are unique; MM processor/prefix cache disabled",
            "- Tail note: p95 is exploratory because each level has 64 requests",
            f"- Attempts: {', '.join(attempt_ids)}; GPU UUIDs: {', '.join(gpu_uuids)}",
            "",
            display_df.to_markdown(index=False),
            "",
            f"Validity: {'PASS' if validity['valid_for_reporting'] else 'CHECK JIT WARNINGS: ' + ', '.join(jit_levels)}",
        ])
        TABLE_PATH.write_text(markdown + "\n", encoding="utf-8")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        colors = {"merged16": "#65758B", "awq": "#D97706"}
        labels = {"merged16": "Merged 16-bit", "awq": "AWQ W4A16 g32"}
        panels = [
            ("output_tokens_per_s", "Output throughput", "tokens/s"),
            ("ttft_p95_ms", "TTFT p95", "ms"),
            ("tpot_p95_ms", "TPOT p95", "ms/token"),
        ]
        for axis, (column, title, ylabel) in zip(axes, panels):
            for tag in ("merged16", "awq"):
                subset = df[df["tag"] == tag].sort_values("concurrency")
                axis.plot(subset["concurrency"], subset[column], marker="o", linewidth=2.2,
                          color=colors[tag], label=labels[tag])
            axis.set_title(title)
            axis.set_xlabel("Concurrency")
            axis.set_ylabel(ylabel)
            axis.set_xticks(CONCURRENCIES)
            axis.grid(alpha=0.25)
        axes[0].legend(frameon=False)
        fig.suptitle("Qwen3-VL ChartQA · A100 · vLLM 0.25.1+cu129 · 64-token fixed decode", y=1.03)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
        plt.show()

        canonical_files = [
            (RESULT_PATH, "bench/benchmark_results.json"),
            (TABLE_PATH, "bench/benchmark_table.md"),
            (PLOT_PATH, "bench/latency_throughput.png"),
            (MANIFEST_PATH, "bench/workload_manifest.json"),
            (Path(gate_path), "bench/quality_gate_source.json"),
        ]
        archive_files = [
            (RESULT_PATH, f"bench/runs/{RUN_ID}/benchmark_results.json"),
            (TABLE_PATH, f"bench/runs/{RUN_ID}/benchmark_table.md"),
            (PLOT_PATH, f"bench/runs/{RUN_ID}/latency_throughput.png"),
            (MANIFEST_PATH, f"bench/runs/{RUN_ID}/workload_manifest.json"),
            (Path(gate_path), f"bench/runs/{RUN_ID}/quality_gate_source.json"),
        ]
        if not validity["valid_for_reporting"]:
            raise RuntimeError(f"benchmark validity gate failed；不發布 canonical 結果：{validity}")
        commit_files_with_retry(
            archive_files + canonical_files,
            f"Publish validated vLLM benchmark {RUN_ID}",
        )

        print("PASS: validity gate 通過；8 組 benchmark 均為 64/64 成功，正式產物已原子 push。")
        print(f"https://huggingface.co/{AWQ_REPO}/tree/main/bench")
        """
    ),
    md(
        r"""
        ## 完成後

        最後一格的表格即為正式結果；結果、圖、manifest 與 checkpoints 都已自動同步到 AWQ 模型 repo 的 `bench/`。若 notebook 以紅色錯誤停止，請依錯誤尾段排查，不要手動跳過該 cell。
        """
    ),
]

for index, cell in enumerate(nb["cells"]):
    cell["id"] = f"stage4-{index:02d}"

nbf.write(nb, "notebooks/benchmark_vllm_colab_cu129.ipynb")
