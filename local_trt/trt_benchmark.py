#!/usr/bin/env python3
"""TRT-LLM benchmarking script.

Single-script solution for running TRT-LLM server in Docker and benchmarking
with Modular's benchmark_serving tool.

Usage:
    ./trt_benchmark.py                          # One-button execution
    ./trt_benchmark.py --tp 8 --concurrency 8   # Custom settings
    ./trt_benchmark.py --nsys --nsys-duration 30  # With profiling
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from config import Config


def log(msg: str) -> None:
    print(f"[trt-bench] {msg}", flush=True)


def nsys_log(msg: str) -> None:
    print(f"[NSYS {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], check: bool = True, capture: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, **kwargs)
    return subprocess.run(cmd, check=check, **kwargs)


def docker_running(container: str) -> bool:
    """Check if a Docker container is running."""
    result = run(["docker", "inspect", "-f", "{{.State.Status}}", container], check=False, capture=True)
    return result.returncode == 0 and result.stdout.strip() == "running"


def docker_exists(container: str) -> bool:
    """Check if a Docker container exists."""
    result = run(["docker", "inspect", container], check=False, capture=True)
    return result.returncode == 0


def wait_for_health(cfg: Config, timeout: int = 600) -> bool:
    """Wait for server to be healthy."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{cfg.port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass

        # Check if container died
        if not docker_running(cfg.container_name):
            log("Container exited unexpectedly!")
            run(["docker", "logs", cfg.container_name], check=False)
            return False

        time.sleep(5)
    return False


def generate_server_config(cfg: Config) -> str:
    """Generate TRT-LLM server YAML config."""
    config = {
        "max_input_len": cfg.max_model_len,
        "cuda_graph_config": {
            "enable_padding": True,
            "max_batch_size": 512,
        },
        "enable_attention_dp": cfg.dp_attention,
        "print_iter_log": True,
        "kv_cache_config": {
            "dtype": "fp8",
            "free_gpu_memory_fraction": 0.8,
            "enable_block_reuse": True,
        },
        "stream_interval": 10,
        "moe_config": {"backend": "TRTLLM"},
        "return_perf_metrics": True,
        "perf_metrics_max_requests": 10000,
        "enable_iter_perf_stats": True,
    }

    if cfg.dp_attention:
        config["attention_dp_config"] = {
            "batching_wait_iters": 0,
            "enable_balance": True,
            "timeout_iters": 60,
        }

    if cfg.mtp_layers > 0:
        config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": cfg.mtp_layers,
        }

    # Write as YAML (simple format, no pyyaml dependency)
    def to_yaml(obj, indent=0) -> str:
        lines = []
        prefix = "    " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    lines.append(to_yaml(v, indent + 1))
                else:
                    val = str(v).lower() if isinstance(v, bool) else v
                    lines.append(f"{prefix}{k}: {val}")
        elif isinstance(obj, list):
            for item in obj:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)

    return to_yaml(config)


def start_server(cfg: Config, config_file: Path) -> Optional[subprocess.Popen]:
    """Start TRT-LLM server in Docker container."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing container
    if docker_exists(cfg.container_name):
        run(["docker", "rm", "-f", cfg.container_name], check=False, capture=True)

    # Build docker command
    max_num_tokens = ((cfg.concurrency + cfg.max_model_len + 64 + 63) // 64) * 64

    server_cmd = [
        "mpirun", "-n", "1", "--oversubscribe", "--allow-run-as-root",
        "trtllm-serve", cfg.model, f"--port={cfg.port}",
        "--trust_remote_code",
        "--backend=pytorch",
        f"--max_seq_len={cfg.max_model_len}",
        f"--max_num_tokens={max_num_tokens}",
        f"--tp_size={cfg.tp}",
        f"--ep_size={cfg.ep_size}",
        f"--extra_llm_api_options=/workspace/config.yml",
    ]

    # Build entrypoint script
    entrypoint_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'hf download "$MODEL"',
    ]

    if cfg.nsys:
        nsys_out = f"/workspace/output/{cfg.result_filename}"
        nsys_cmd = [
            "nsys", "profile",
            f"--output={nsys_out}",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
        ]
        if cfg.nsys_delay > 0:
            nsys_cmd.extend(["--delay", str(cfg.nsys_delay)])
        if cfg.nsys_duration > 0:
            nsys_cmd.extend(["--duration", str(cfg.nsys_duration)])
        entrypoint_lines.append(" ".join(nsys_cmd + server_cmd))
    else:
        entrypoint_lines.append(" ".join(server_cmd))

    entrypoint_script = "\n".join(entrypoint_lines)

    # Detect nvidia runtime
    result = run(["docker", "info", "--format", "{{json .Runtimes}}"], capture=True, check=False)
    runtime_args = ["--runtime", "nvidia"] if '"nvidia"' in result.stdout else []

    # Host library mounts for non-nvidia runtime
    host_lib_mounts = []
    host_lib_env = []
    if not runtime_args:
        host_lib_dir = "/opt/host-libs"
        libs = [
            ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "libcuda.so.1"),
            ("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "libnvidia-ml.so.1"),
            ("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1", "libnvidia-ptxjitcompiler.so.1"),
        ]
        for host_path, name in libs:
            if os.path.exists(host_path):
                host_lib_mounts.extend(["-v", f"{host_path}:{host_lib_dir}/{name}:ro"])
        if host_lib_mounts:
            host_lib_env = ["-e", f"LD_LIBRARY_PATH={host_lib_dir}"]

    docker_cmd = [
        "docker", "run", "--init", "--network", "host", "--detach",
        "--name", cfg.container_name,
        *runtime_args,
        "--gpus", "all", "--ipc", "host", "--privileged",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{Path(__file__).parent}:/workspace",
        "-v", f"{cfg.hf_cache}:/root/.cache/huggingface",
        "-v", f"{cfg.output_dir}:/workspace/output",
        "-v", f"{config_file}:/workspace/config.yml:ro",
        *host_lib_mounts,
        *host_lib_env,
        "-e", f"HF_TOKEN={os.environ['HF_TOKEN']}",
        "-e", "HF_HUB_CACHE=/root/.cache/huggingface",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "NCCL_GRAPH_REGISTER=0",
        "-e", f"MODEL={cfg.model}",
        "--entrypoint", "/bin/bash",
        cfg.image,
        "-c", entrypoint_script,
    ]

    log(f"Starting container {cfg.container_name}...")
    run(docker_cmd, check=True)
    return None  # Container runs detached


def stop_server(cfg: Config) -> None:
    """Stop the TRT-LLM server container."""
    if docker_exists(cfg.container_name):
        log(f"Stopping container {cfg.container_name}...")
        run(["docker", "rm", "-f", cfg.container_name], check=False, capture=True)


def run_benchmark(cfg: Config) -> int:
    """Run benchmark_serving via bazel."""
    bazelw = cfg.modular_repo / "bazelw"
    if not bazelw.exists():
        log(f"ERROR: bazelw not found at {bazelw}")
        return 1

    cmd = [
        str(bazelw), "run", "--config=disable-mypy",
        "//max/python/max/benchmark:benchmark_serving", "--",
        "--backend", "trtllm-chat",
        "--host", "localhost",
        "--port", str(cfg.port),
        "--dataset-name", "random",
        "--model", cfg.model,
        "--request-rate", "inf",
        "--num-prompts", str(cfg.num_prompts),
        "--random-input-len", str(cfg.input_len),
        "--random-output-len", str(cfg.output_len),
        "--max-concurrency", str(cfg.concurrency),
        "--random-sys-prompt-ratio", "0.96",
        "--random-distribution-type", "uniform",
        "--collect-gpu-stats",
        "--skip-test-prompt",
    ]

    log("Running benchmark...")
    result = run(cmd, check=False, cwd=cfg.modular_repo)
    return result.returncode


def scrape_metrics(cfg: Config) -> None:
    """Scrape performance metrics from server."""
    import urllib.request
    import urllib.error

    base_url = f"http://localhost:{cfg.port}"

    for endpoint, suffix in [("/perf_metrics", ".perf_metrics.json"), ("/metrics", ".iter_metrics.json")]:
        try:
            with urllib.request.urlopen(f"{base_url}{endpoint}", timeout=5) as resp:
                data = resp.read()
                output_file = cfg.output_dir / f"{cfg.result_filename}{suffix}"
                output_file.write_bytes(data)
                log(f"Saved {output_file.name}")
        except (urllib.error.URLError, TimeoutError) as e:
            log(f"Failed to scrape {endpoint}: {e}")


def export_nsys_sqlite(cfg: Config) -> None:
    """Export NSYS report to SQLite."""
    nsys_rep = cfg.nsys_output
    if not nsys_rep.exists():
        log(f"NSYS report not found: {nsys_rep}")
        return

    sqlite_out = nsys_rep.with_suffix(".sqlite")
    nsys_log(f"Exporting to SQLite: {sqlite_out}")
    run([
        "docker", "exec", cfg.container_name,
        "nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
        "--output", f"/workspace/output/{sqlite_out.name}",
        f"/workspace/output/{nsys_rep.name}",
    ], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRT-LLM benchmarking tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument("--model", default="nvidia/DeepSeek-R1-0528-FP4-V2", help="HuggingFace model path")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--dp-attention", action="store_true", help="Enable DP attention")
    parser.add_argument("--max-model-len", type=int, default=160000, help="Maximum sequence length")
    parser.add_argument("--mtp-layers", type=int, default=1, help="MTP layers (0=disabled, 1-5=enabled)")

    # Benchmark settings
    parser.add_argument("--num-prompts", type=int, default=300, help="Number of prompts")
    parser.add_argument("--input-len", type=int, default=75000, help="Random input length")
    parser.add_argument("--output-len", type=int, default=300, help="Random output length")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent requests")

    # NSYS profiling
    parser.add_argument("--nsys", action="store_true", help="Enable NSYS profiling")
    parser.add_argument("--nsys-duration", type=int, default=0, help="NSYS capture duration (0=until end)")
    parser.add_argument("--nsys-delay", type=int, default=0, help="NSYS capture delay")
    parser.add_argument("--nsys-sqlite", action="store_true", help="Export NSYS to SQLite")

    # Runtime flags
    parser.add_argument("--skip-server", action="store_true", help="Skip server startup (use existing)")

    args = parser.parse_args()

    # Build config
    cfg = Config(
        model=args.model,
        tp=args.tp,
        ep_size=args.ep_size,
        dp_attention=args.dp_attention,
        max_model_len=args.max_model_len,
        mtp_layers=args.mtp_layers,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        concurrency=args.concurrency,
        nsys=args.nsys,
        nsys_duration=args.nsys_duration,
        nsys_delay=args.nsys_delay,
        nsys_sqlite=args.nsys_sqlite,
        skip_server=args.skip_server,
    )

    try:
        cfg.validate()
    except ValueError as e:
        log(f"ERROR: {e}")
        return 1

    log(f"Config: {cfg}")

    # Setup signal handler for cleanup
    def cleanup(signum=None, frame=None):
        if not cfg.skip_server:
            stop_server(cfg)
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Generate server config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(generate_server_config(cfg))
        config_file = Path(f.name)

    try:
        if cfg.skip_server:
            log("Skipping server startup (--skip-server)")
            # Quick health check
            import urllib.request
            try:
                with urllib.request.urlopen(f"http://localhost:{cfg.port}/health", timeout=2):
                    log("Server is healthy")
            except Exception as e:
                log(f"ERROR: Server not healthy: {e}")
                return 1
        else:
            # Start server
            start_server(cfg, config_file)

            # Wait for health
            log("Waiting for server to be ready...")
            if not wait_for_health(cfg):
                log("ERROR: Server failed to become healthy")
                return 1
            log("Server is ready!")

        # Run benchmark
        bench_rc = run_benchmark(cfg)

        # Scrape metrics
        scrape_metrics(cfg)

        # Export NSYS SQLite if requested
        if cfg.nsys and cfg.nsys_sqlite:
            export_nsys_sqlite(cfg)

        if bench_rc != 0:
            log(f"Benchmark failed with exit code {bench_rc}")
        else:
            log("Benchmark completed successfully!")

        return bench_rc

    finally:
        config_file.unlink(missing_ok=True)
        # Keep container running for inspection (matches old behavior)
        log(f"Container {cfg.container_name} left running for inspection")
        log(f"  View logs: docker logs -f {cfg.container_name}")
        log(f"  Stop: docker rm -f {cfg.container_name}")


if __name__ == "__main__":
    sys.exit(main())
