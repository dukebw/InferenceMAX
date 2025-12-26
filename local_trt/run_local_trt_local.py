#!/usr/bin/env python3

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _wait_for_server_ready(
    port: int, server_log: Path, server_proc: subprocess.Popen, sleep_interval: int = 5
) -> None:
    with server_log.open("r", encoding="utf-8", errors="ignore") as log_f:
        while True:
            line = log_f.readline()
            if line:
                print(line.rstrip())
            try:
                with urlopen(f"http://0.0.0.0:{port}/health", timeout=2):
                    return
            except URLError:
                if server_proc.poll() is not None:
                    log_f.seek(0)
                    lines = log_f.readlines()[-200:]
                    if lines:
                        print("---- server log tail ----")
                        for tail_line in lines:
                            print(tail_line.rstrip())
                        print("---- end server log tail ----")
                    raise RuntimeError("Server died before becoming healthy.")
                time.sleep(sleep_interval)


def _fetch_json(url: str):
    try:
        with urlopen(url, timeout=5) as resp:
            payload = resp.read()
        if not payload:
            return None
        return json.loads(payload)
    except Exception:
        return None


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _summarize_spec_decoding(iter_stats: list, out_path: Path) -> None:
    total_draft = 0
    total_accepted = 0
    total_req_with_draft = 0
    weighted_accept_len = 0.0
    iters_with_spec = 0

    def parse_entry(entry):
        if isinstance(entry, str):
            try:
                return json.loads(entry)
            except json.JSONDecodeError:
                return None
        if isinstance(entry, dict):
            return entry
        return None

    for entry in iter_stats if isinstance(iter_stats, list) else []:
        parsed = parse_entry(entry)
        if not parsed:
            continue
        spec = parsed.get("specDecodingStats") or parsed.get("spec_decoding_stats")
        if not isinstance(spec, dict):
            continue
        draft = int(spec.get("numDraftTokens", 0))
        accepted = int(spec.get("numAcceptedTokens", 0))
        reqs = int(spec.get("numRequestsWithDraftTokens", 0))
        accept_len = float(spec.get("acceptanceLength", 0.0))
        if draft <= 0:
            continue
        iters_with_spec += 1
        total_draft += draft
        total_accepted += accepted
        total_req_with_draft += reqs
        weighted_accept_len += accept_len * reqs

    acceptance_rate = total_accepted / total_draft if total_draft > 0 else None
    avg_accept_len = (
        weighted_accept_len / total_req_with_draft if total_req_with_draft > 0 else None
    )

    summary = {
        "total_draft_tokens": total_draft,
        "total_accepted_tokens": total_accepted,
        "acceptance_rate": acceptance_rate,
        "avg_acceptance_length": avg_accept_len,
        "iterations_with_spec": iters_with_spec,
        "requests_with_draft_tokens": total_req_with_draft,
    }
    _write_json(out_path, summary)
    if acceptance_rate is None:
        print("Speculative acceptance rate unavailable (no draft tokens).")
    else:
        print(
            f"Speculative acceptance rate: {acceptance_rate:.4f} "
            f"({total_accepted}/{total_draft})"
        )
        if avg_accept_len is not None:
            print(f"Avg acceptance length: {avg_accept_len:.3f}")


def _run_benchmark_serving(
    model: str,
    port: int,
    input_len: int,
    output_len: int,
    random_range_ratio: float,
    num_prompts: int,
    max_concurrency: int,
    result_filename: str,
    result_dir: Path,
) -> None:
    bench_dir = Path(tempfile.mkdtemp(prefix="bmk-"))
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/kimbochen/bench_serving.git",
            str(bench_dir),
        ],
        check=True,
    )
    cmd = [
        sys.executable,
        str(bench_dir / "benchmark_serving.py"),
        "--model",
        model,
        "--backend",
        "openai",
        "--base-url",
        f"http://0.0.0.0:{port}",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(random_range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--save-result",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        f"{result_filename}.json",
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(os.getenv("OUTPUT_DIR", str(script_dir / "output")))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = os.getenv("MODEL", "nvidia/DeepSeek-R1-0528-FP4-V2")
    tp = _env_int("TP", 8)
    ep_size = _env_int("EP_SIZE", 1)
    dp_attention = os.getenv("DP_ATTENTION", "false")
    conc = _env_int("CONC", 4)
    isl = _env_int("ISL", 1024)
    osl = _env_int("OSL", 1024)
    max_model_len = _env_int("MAX_MODEL_LEN", 2248)
    random_range_ratio = _env_float("RANDOM_RANGE_RATIO", 0.8)
    port = _env_int("PORT", 8888)
    result_filename = os.getenv(
        "RESULT_FILENAME",
        f"dsr1_1k1k_fp4_trt_tp{tp}_ep{ep_size}_dpa_{dp_attention}_conc{conc}_local",
    )

    num_prompts = os.getenv("NUM_PROMPTS")
    num_prompts_multiplier = _env_int("NUM_PROMPTS_MULTIPLIER", 10)
    if not num_prompts:
        num_prompts = str(conc * num_prompts_multiplier)
    num_prompts_int = int(num_prompts)

    mtp_enable = _env_bool("MTP_ENABLE", True)
    mtp_num_nextn_predict_layers = _env_int("MTP_NUM_NEXTN_PREDICT_LAYERS", 1)
    scrape_metrics = _env_bool("SCRAPE_METRICS", True)

    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN is required to download the model.")

    moe_backend = "TRTLLM"
    if tp == 4:
        if isl == 1024 and osl == 1024 and conc >= 256:
            moe_backend = "CUTLASS"
        elif isl == 1024 and osl == 8192 and conc >= 256:
            moe_backend = "CUTLASS"
        elif isl == 8192 and osl == 1024 and conc > 32:
            moe_backend = "CUTLASS"
    elif tp == 8:
        if isl == 1024 and osl == 1024 and conc >= 256:
            moe_backend = "CUTLASS"
        elif isl == 1024 and osl == 8192 and conc >= 256:
            moe_backend = "CUTLASS"
        elif isl == 8192 and osl == 1024 and conc > 32:
            moe_backend = "CUTLASS"

    print(f"MOE_BACKEND set to '{moe_backend}'")

    max_num_tokens = ((conc + isl + 64 + 63) // 64) * 64

    extra_cfg = textwrap.dedent(
        f"""
        cuda_graph_config:
            enable_padding: true
            max_batch_size: 512
        enable_attention_dp: {dp_attention}
        print_iter_log: true
        kv_cache_config:
            dtype: fp8
            free_gpu_memory_fraction: 0.8
            enable_block_reuse: true
        stream_interval: 10
        moe_config:
            backend: {moe_backend}
        return_perf_metrics: true
        perf_metrics_max_requests: 10000
        enable_iter_perf_stats: true
        """
    )

    if dp_attention.lower() == "true":
        extra_cfg += textwrap.dedent(
            """
            attention_dp_config:
                batching_wait_iters: 0
                enable_balance: true
                timeout_iters: 60
            """
        )

    if mtp_enable:
        extra_cfg += textwrap.dedent(
            f"""
            speculative_config:
                decoding_type: MTP
                num_nextn_predict_layers: {mtp_num_nextn_predict_layers}
            """
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_cfg:
        tmp_cfg.write(extra_cfg)
        extra_cfg_path = tmp_cfg.name

    server_log = Path(tempfile.mkstemp(prefix="trtllm-server-", suffix=".log")[1])

    if shutil.which("hf") is not None:
        subprocess.run(["hf", "download", model], check=True)
    else:
        print("WARNING: 'hf' CLI not found; skipping hf download step.")

    cmd = []
    use_mpirun = _env_bool("USE_MPIRUN", True)
    if use_mpirun and shutil.which("mpirun"):
        cmd = ["mpirun", "-n", "1", "--oversubscribe"]
        if os.geteuid() == 0:
            cmd.append("--allow-run-as-root")

    entrypoint = os.getenv("TRTLLM_ENTRYPOINT", "")
    server_python = os.getenv("TRTLLM_PYTHON")
    if server_python:
        cmd += [server_python, "-m", "tensorrt_llm.commands.serve", model]
        print(f"Using TRTLLM_PYTHON={server_python}")
    elif entrypoint or shutil.which("trtllm-serve"):
        cmd += ["trtllm-serve", model]
        print("Using trtllm-serve from PATH.")
    else:
        cmd += [sys.executable, "-m", "tensorrt_llm.commands.serve", model]
        print("Using current Python to launch TRT-LLM (set TRTLLM_PYTHON to override).")

    cmd += [
        "--port",
        str(port),
        "--trust_remote_code",
        "--backend=pytorch",
        "--max_seq_len",
        str(max_model_len),
        "--max_num_tokens",
        str(max_num_tokens),
        "--tp_size",
        str(tp),
        "--ep_size",
        str(ep_size),
        "--extra_llm_api_options",
        extra_cfg_path,
    ]

    print("Launching server:", " ".join(cmd))
    print(f"Server log: {server_log}")
    with server_log.open("w", encoding="utf-8") as log_f:
        server_proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    def _shutdown(_signum=None, _frame=None):
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        _wait_for_server_ready(port, server_log, server_proc)
        _run_benchmark_serving(
            model=model,
            port=port,
            input_len=isl,
            output_len=osl,
            random_range_ratio=random_range_ratio,
            num_prompts=num_prompts_int,
            max_concurrency=conc,
            result_filename=result_filename,
            result_dir=output_dir,
        )

        if scrape_metrics:
            base_url = f"http://0.0.0.0:{port}"
            perf_metrics = _fetch_json(f"{base_url}/perf_metrics")
            iter_metrics = _fetch_json(f"{base_url}/metrics")
            if perf_metrics is not None:
                _write_json(
                    output_dir / f"{result_filename}.perf_metrics.json", perf_metrics
                )
            if iter_metrics is not None:
                _write_json(
                    output_dir / f"{result_filename}.iter_metrics.json", iter_metrics
                )
                _summarize_spec_decoding(
                    iter_metrics, output_dir / f"{result_filename}.spec_decoding.json"
                )
    finally:
        _shutdown()
        if server_proc.poll() is None:
            server_proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
