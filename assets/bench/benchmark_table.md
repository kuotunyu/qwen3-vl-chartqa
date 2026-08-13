# Qwen3-VL ChartQA vLLM benchmark

- Run ID: `v2-aa4442870cfd`
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB
- Quality gate: PASS (86.24% → 85.52%, -0.72pp)
- Workload: 64 real ChartQA requests per level; fixed 64 output tokens; ignore EOS
- Fairness: same 64-source cohort at every concurrency; per-level variants preserve dimensions/prompts
- Token workload: merged/AWQ input_lens are identical at all four concurrency levels
- Cache control: encoded and decoded-pixel hashes are unique; MM processor/prefix cache disabled
- Tail note: p95 is exploratory because each level has 64 requests
- Attempts: b5bd63eafe31; GPU UUIDs: GPU-4be520cb-f1f8-0b6e-297b-579d54c9ed52

| model         |   concurrency |   n |   failed |   requests_per_s |   output_tokens_per_s |   ttft_p50_ms |   ttft_p95_ms |   tpot_p50_ms |   tpot_p95_ms |   e2e_p50_ms |   e2e_p95_ms |
|:--------------|--------------:|----:|---------:|-----------------:|----------------------:|--------------:|--------------:|--------------:|--------------:|-------------:|-------------:|
| merged-16bit  |             1 |  64 |        0 |             1.05 |                 67.29 |        101.85 |        160.76 |         13.34 |         13.43 |       942.22 |      1007.16 |
| merged-16bit  |             4 |  64 |        0 |             3.61 |                231.02 |        218.78 |        299.78 |         13.89 |         15.04 |      1097.74 |      1169.61 |
| merged-16bit  |             8 |  64 |        0 |             6.06 |                387.58 |        310.54 |        473.68 |         16.02 |         18.33 |      1325.89 |      1426.9  |
| merged-16bit  |            16 |  64 |        0 |             9.3  |                595.06 |        468.69 |        806.77 |         19.27 |         24.14 |      1657.09 |      2005.45 |
| awq-w4a16-g32 |             1 |  64 |        0 |             1.93 |                123.24 |        110.67 |        154.81 |          6.36 |          6.47 |       511.64 |       562    |
| awq-w4a16-g32 |             4 |  64 |        0 |             5.58 |                356.95 |        223.22 |        326.64 |          7.6  |          9.11 |       708.33 |       776.13 |
| awq-w4a16-g32 |             8 |  64 |        0 |             8.26 |                528.48 |        359.46 |        572.66 |          9.58 |         12.4  |       954.82 |      1038.69 |
| awq-w4a16-g32 |            16 |  64 |        0 |            10.97 |                701.9  |        564.26 |        958.28 |         14.5  |         19.93 |      1418.43 |      1612.51 |

Validity: PASS
