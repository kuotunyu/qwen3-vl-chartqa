# Qwen3-VL 8B ChartQA GGUF

Fine-tuned GGUF artifacts for CPU inference through pinned `llama.cpp`.

## Artifact identity

- Repository: `steven0226/qwen3vl-8b-chartqa-gguf`
- Revision: `5e5860f5d4060f87614ecfb6243a001067979276`
- Source merged revision: `519060ef43df3261e0512e5ae4c82a4d4e675f32`
- llama.cpp commit: `79bba02a6741de194912d370015866414faa83ad`

| File | Quantization | Bytes | SHA-256 |
|---|---|---:|---|
| `Qwen3VL-8B-ChartQA-Q4_K_M.gguf` | Q4_K_M | 5,027,784,800 | `9419649d680756ffd6a03b0817130b6a42276c08376e85e57c82e3fa57ce0cc8` |
| `mmproj-Qwen3VL-8B-ChartQA-Q8_0.gguf` | Q8_0 | 752,289,728 | `b196c5f7504c06855ae6e08e78f76173a63423e58074c1e35661a2b96b21d048` |

## Verification

The conversion notebook ran an independent CPU multimodal smoke test using one pinned ChartQA sample fetched at runtime. The inference result passed an exact-match check against the runtime label; the image, question, label, and model output are not published.

This smoke test verifies conversion integrity and multimodal execution. The canonical 2,500-question accuracy result remains the AWQ/vLLM evaluation (85.52% overall); results from different inference stacks are not mixed.

## Deployment note

The Docker CPU Space source is ready, but the authenticated API requires PRO for this account and no paid resource was created. The completed no-cost deployment is the [persistent Static Space portfolio](https://huggingface.co/spaces/steven0226/qwen3vl-chartqa-demo), which links to this artifact and opens the hosted interactive notebook in Colab A100.
