# TRT-LLM Benchmark

Single-script solution for running TRT-LLM server in Docker and benchmarking with Modular's `benchmark_serving` tool.

## Quick Start

```bash
cd /home/ubuntu/work/InferenceMAX/local_trt
export HF_TOKEN=...

# One-button execution (all defaults)
./trt_benchmark.py

# With NSYS profiling
./trt_benchmark.py --nsys

# Custom settings
./trt_benchmark.py --tp 8 --concurrency 8 --mtp-layers 3
```

## CLI Options

```
--model MODEL           HuggingFace model (default: nvidia/DeepSeek-R1-0528-FP4-V2)
--tp TP                 Tensor parallelism (default: 8)
--ep-size EP_SIZE       Expert parallelism (default: 1)
--dp-attention          Enable DP attention
--max-model-len LEN     Max sequence length (default: 160000)
--mtp-layers N          MTP layers, 0=disabled (default: 1)

--num-prompts N         Number of prompts (default: 300)
--input-len N           Random input length (default: 75000)
--output-len N          Random output length (default: 300)
--concurrency N         Max concurrent requests (default: 8)

--nsys                  Enable NSYS profiling
--nsys-duration SEC     Capture duration, 0=until end (default: 0)
--nsys-delay SEC        Capture delay after server ready (default: 0)
--nsys-sqlite           Export NSYS report to SQLite

--skip-server           Use existing server (skip container startup)
```

## Output Files

All outputs in `./output/`:

| File | Description |
|------|-------------|
| `*.nsys-rep` | NSYS trace (if `--nsys`) |
| `*.sqlite` | NSYS SQLite export (if `--nsys-sqlite`) |
| `*.perf_metrics.json` | Per-request TRT-LLM metrics |
| `*.iter_metrics.json` | Per-iteration server stats |

## Examples

```bash
# MTP sweep (0-5 layers)
for n in 0 1 2 3 4 5; do
  ./trt_benchmark.py --mtp-layers $n --num-prompts 100
done

# Profile with 30s capture after 10s warmup
./trt_benchmark.py --nsys --nsys-delay 10 --nsys-duration 30

# Benchmark against existing server
./trt_benchmark.py --skip-server --concurrency 16
```

## Requirements

- Docker with NVIDIA GPU support
- `HF_TOKEN` environment variable
- Modular repo at `~/work/modular` (for `benchmark_serving`)
