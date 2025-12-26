# Local TRT-LLM Benchmark (InferenceMAX-style)

Runs DeepSeek-R1 FP4 on B200 in the TensorRT-LLM container with the same server
flags/config style used by InferenceMAX, plus optional Nsight Systems tracing.

## Quick start
```bash
cd /home/ubuntu/work/InferenceMAX/local_trt
export HF_TOKEN=...
# Benchmark only
./launch_local_trt.sh
# Benchmark + nsys trace (server only)
NSYS_PROFILE=1 ./launch_local_trt.sh
```

## Output locations
All outputs are written to `OUTPUT_DIR` on the host (default:
`/home/ubuntu/work/InferenceMAX/local_trt/output`).

- Benchmark JSON: `${OUTPUT_DIR}/${RESULT_FILENAME}.json`
- Nsight Systems trace: `${OUTPUT_DIR}/${RESULT_FILENAME}.nsys-rep`
- TRT-LLM perf metrics: `${OUTPUT_DIR}/${RESULT_FILENAME}.perf_metrics.json`
- TRT-LLM iteration stats: `${OUTPUT_DIR}/${RESULT_FILENAME}.iter_metrics.json`
- Spec-decoding summary: `${OUTPUT_DIR}/${RESULT_FILENAME}.spec_decoding.json`

If `NSYS_EXPORT_SQLITE=1`, a SQLite export will be written to
`${OUTPUT_DIR}/${RESULT_FILENAME}.sqlite`.

## Defaults
- Model: `nvidia/DeepSeek-R1-0528-FP4-V2`
- TP=8, EP=1, CONC=4, DP_ATTENTION=false
- ISL=1024, OSL=1024, MAX_MODEL_LEN=2248
- MTP enabled (num_nextn_predict_layers=1)

## Spec-decoding acceptance rate
The request-level `/perf_metrics` endpoint does not always report draft token
counts for MTP. This harness also scrapes `/metrics` (iteration stats) and
computes an aggregate acceptance rate from `specDecodingStats`:

- `numAcceptedTokens / numDraftTokens` (aggregate)
- average `acceptanceLength` (weighted by `numRequestsWithDraftTokens`)

The computed summary is written to:
`${OUTPUT_DIR}/${RESULT_FILENAME}.spec_decoding.json`

## Environment variables to tune
**Core**
- `TP` (default 8)
- `EP_SIZE` (default 1)
- `CONC` (default 4)
- `DP_ATTENTION` (default false)
- `ISL`, `OSL` (default 1024/1024)
- `MAX_MODEL_LEN` (default 2248)
- `RANDOM_RANGE_RATIO` (default 0.8)
- `NUM_PROMPTS` (explicit prompt count; overrides multiplier)
- `NUM_PROMPTS_MULTIPLIER` (default 10; used when `NUM_PROMPTS` is unset)

**MTP (multi-token prediction)**
- `MTP_ENABLE` (default 1)
- `MTP_NUM_NEXTN_PREDICT_LAYERS` (default 1)
  - These two must match (per TRT-LLM MTP requirements).

**Tracing**
- `NSYS_PROFILE` (default 0)
- `NSYS_TRACE` (default `cuda,nvtx,osrt`)
- `NSYS_CUDA_GRAPH_TRACE` (default `node`; use `graph` for lower overhead)
- `NSYS_DELAY_SEC` (default 0; delay after server is ready before profiling starts)
- `NSYS_DURATION_SEC` (default 0; if >0, capture runs async for this many seconds)
- `NSYS_OUT` (default `/workspace/output/${RESULT_FILENAME}` inside the container;
  ends up as `${OUTPUT_DIR}/${RESULT_FILENAME}.nsys-rep` on host)
- `NSYS_EXPORT_SQLITE` (default 0)
- `NSYS_START_BEFORE_HEALTH` (default 0; use if your nsys build requires `start`
  before health becomes ready; ignored when `NSYS_DURATION_SEC` > 0)

**Metrics**
- `SCRAPE_METRICS` (default 1; scrapes `/perf_metrics` and `/metrics`)

**Paths**
- `OUTPUT_DIR` (default `local_trt/output`)
- `HF_CACHE` (default `~/.cache/huggingface`)

## Notes
- Requires Docker with NVIDIA GPU support and a valid `HF_TOKEN`.
- `nsys` is run inside the TRT-LLM container; the trace is renamed to your
  `${RESULT_FILENAME}` when the run finishes.
- If `NSYS_DURATION_SEC` > 0, profiling starts after `NSYS_DELAY_SEC` (post
  readiness) and stops asynchronously while the benchmark continues.
