#!/usr/bin/env bash

set -euo pipefail

PORT="${PORT:-8888}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_SESSION="${NSYS_SESSION:-trtllm-profile}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_OUT="${NSYS_OUT:-/workspace/output/${RESULT_FILENAME:-trtllm-trace}}"
NSYS_EXPORT_SQLITE="${NSYS_EXPORT_SQLITE:-0}"
NSYS_START_BEFORE_HEALTH="${NSYS_START_BEFORE_HEALTH:-0}"
NSYS_DELAY_SEC="${NSYS_DELAY_SEC:-0}"
NSYS_DURATION_SEC="${NSYS_DURATION_SEC:-0}"
MTP_ENABLE="${MTP_ENABLE:-1}"
MTP_NUM_NEXTN_PREDICT_LAYERS="${MTP_NUM_NEXTN_PREDICT_LAYERS:-1}"
SCRAPE_METRICS="${SCRAPE_METRICS:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
NSYS_REPORT_DIR="$(dirname "$NSYS_OUT")"

source "/workspace/benchmarks/benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO RESULT_FILENAME EP_SIZE DP_ATTENTION

hf download "$MODEL"

mkdir -p "$NSYS_REPORT_DIR"
cd "$NSYS_REPORT_DIR"

MOE_BACKEND="TRTLLM"

if [[ "$TP" == "4" ]]; then
  if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    if [[ $CONC -ge 256 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    if [[ $CONC -ge 256 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ $CONC -gt 32 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  fi
elif [[ "$TP" == "8" ]]; then
  if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    if [[ $CONC -ge 256 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    if [[ $CONC -ge 256 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ $CONC -gt 32 ]]; then
      MOE_BACKEND="CUTLASS"
    fi
  fi
fi

echo "MOE_BACKEND set to '$MOE_BACKEND'"

if [[ "$NSYS_PROFILE" == "1" ]]; then
  echo "=== NSYS Profiling Enabled ==="
  echo "  Session:        $NSYS_SESSION"
  echo "  Trace:          $NSYS_TRACE"
  echo "  Output:         $NSYS_OUT"
  echo "  Delay:          ${NSYS_DELAY_SEC}s"
  echo "  Duration:       ${NSYS_DURATION_SEC}s (0=until benchmark ends)"
  echo "  Start before health: $NSYS_START_BEFORE_HEALTH"
  echo "=============================="
fi

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
EXTRA_CONFIG_FILE=$(mktemp /tmp/dsr1-fp4-XXXXXX.yml)

cat > "$EXTRA_CONFIG_FILE" << EOF
max_input_len: $MAX_MODEL_LEN
cuda_graph_config:
    enable_padding: true
    max_batch_size: 512
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: true
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
return_perf_metrics: true
perf_metrics_max_requests: 10000
enable_iter_perf_stats: true
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
  cat << EOF >> "$EXTRA_CONFIG_FILE"
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
fi

if [[ "$MTP_ENABLE" == "1" ]]; then
  cat << EOF >> "$EXTRA_CONFIG_FILE"
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $MTP_NUM_NEXTN_PREDICT_LAYERS
EOF
fi

MAX_NUM_TOKENS=$(( (CONC + MAX_MODEL_LEN + 64 + 63) / 64 * 64 ))

TRTLLM_CMD=(
  mpirun -n 1 --oversubscribe --allow-run-as-root
  trtllm-serve "$MODEL" --port="$PORT"
  --trust_remote_code
  --backend=pytorch
  --max_seq_len="$MAX_MODEL_LEN"
  --max_num_tokens="$MAX_NUM_TOKENS"
  --tp_size="$TP" --ep_size="$EP_SIZE"
  --extra_llm_api_options="$EXTRA_CONFIG_FILE"
)

fetch_perf_metrics() {
  local base_url="http://0.0.0.0:${PORT}"
  local perf_file="/workspace/output/${RESULT_FILENAME}.perf_metrics.json"
  local iter_file="/workspace/output/${RESULT_FILENAME}.iter_metrics.json"
  local spec_file="/workspace/output/${RESULT_FILENAME}.spec_decoding.json"

  set +e
  curl -s "${base_url}/perf_metrics" -o "$perf_file"
  curl -s "${base_url}/metrics" -o "$iter_file"
  set -e

  python3 - "$iter_file" "$spec_file" <<'PY'
import json
import os
import sys

iter_path = sys.argv[1]
out_path = sys.argv[2]

if not os.path.exists(iter_path) or os.path.getsize(iter_path) == 0:
    print("No iteration metrics available for speculative decoding.")
    sys.exit(0)

with open(iter_path, "r", encoding="utf-8") as f:
    try:
        raw = json.load(f)
    except json.JSONDecodeError:
        print("Iteration metrics were not valid JSON.")
        sys.exit(0)

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

for entry in raw if isinstance(raw, list) else []:
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

acceptance_rate = (
    total_accepted / total_draft if total_draft > 0 else None
)
avg_accept_len = (
    weighted_accept_len / total_req_with_draft
    if total_req_with_draft > 0
    else None
)

summary = {
    "total_draft_tokens": total_draft,
    "total_accepted_tokens": total_accepted,
    "acceptance_rate": acceptance_rate,
    "avg_acceptance_length": avg_accept_len,
    "iterations_with_spec": iters_with_spec,
    "requests_with_draft_tokens": total_req_with_draft,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

if acceptance_rate is None:
    print("Speculative acceptance rate unavailable (no draft tokens).")
else:
    print(f"Speculative acceptance rate: {acceptance_rate:.4f} ({total_accepted}/{total_draft})")
    if avg_accept_len is not None:
        print(f"Avg acceptance length: {avg_accept_len:.3f}")
PY
}

nsys_log() {
  echo "[NSYS $(date '+%H:%M:%S')] $*"
}

finalize_nsys_report() {
  nsys_log "Finalizing report..."
  local report_file=""
  local target=""

  if ls "$NSYS_REPORT_DIR"/*.nsys-rep >/dev/null 2>&1; then
    report_file=$(ls -t "$NSYS_REPORT_DIR"/*.nsys-rep | head -n 1)
  elif ls "$NSYS_REPORT_DIR"/*.qdrep >/dev/null 2>&1; then
    report_file=$(ls -t "$NSYS_REPORT_DIR"/*.qdrep | head -n 1)
  fi

  if [[ -z "$report_file" ]]; then
    nsys_log "No report file found"
    return 0
  fi

  local ext="${report_file##*.}"
  if [[ "$NSYS_OUT" == *.nsys-rep || "$NSYS_OUT" == *.qdrep ]]; then
    target="$NSYS_OUT"
  else
    target="${NSYS_OUT}.${ext}"
  fi

  if [[ "$report_file" != "$target" ]]; then
    mv -f "$report_file" "$target"
  fi
  nsys_log "Report saved: $target"

  if [[ "$NSYS_EXPORT_SQLITE" == "1" ]]; then
    nsys_log "Exporting to SQLite..."
    nsys export --type sqlite --force-overwrite true \
      --output "${NSYS_OUT}.sqlite" "$target" || true
    nsys_log "SQLite export: ${NSYS_OUT}.sqlite"
  fi
}

start_async_nsys_capture() {
  nsys_log "Async capture scheduled: delay=${NSYS_DELAY_SEC}s, duration=${NSYS_DURATION_SEC}s"
  (
    if [[ "$NSYS_DELAY_SEC" -gt 0 ]]; then
      nsys_log "Waiting ${NSYS_DELAY_SEC}s before capture..."
      sleep "$NSYS_DELAY_SEC"
    fi
    nsys_log ">>> CAPTURE START <<<"
    nsys start --session="$NSYS_SESSION"
    nsys_log "Capturing for ${NSYS_DURATION_SEC}s..."
    sleep "$NSYS_DURATION_SEC"
    nsys_log ">>> CAPTURE STOP <<<"
    nsys stop --session="$NSYS_SESSION"
    finalize_nsys_report
  ) &
  NSYS_ASYNC_PID=$!
}

set -x

if [[ "$NSYS_PROFILE" == "1" ]]; then
  nsys_log "Launching server with profiling (session=$NSYS_SESSION, trace=$NSYS_TRACE)"
  nsys launch \
    --session-new="$NSYS_SESSION" \
    --trace="$NSYS_TRACE" \
    --cuda-graph-trace="$NSYS_CUDA_GRAPH_TRACE" \
    "${TRTLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
else
  "${TRTLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
fi

SERVER_PID=$!

if [[ "$NSYS_PROFILE" == "1" && "$NSYS_DURATION_SEC" -le 0 && "$NSYS_START_BEFORE_HEALTH" == "1" ]]; then
  nsys_log ">>> CAPTURE START (before health check) <<<"
  nsys start --session="$NSYS_SESSION"
fi

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$NSYS_PROFILE" == "1" ]]; then
  if [[ "$NSYS_DURATION_SEC" -gt 0 ]]; then
    start_async_nsys_capture
  elif [[ "$NSYS_START_BEFORE_HEALTH" != "1" ]]; then
    nsys_log ">>> CAPTURE START (after health check) <<<"
    nsys start --session="$NSYS_SESSION"
  fi
fi

if [[ "$RUN_BENCHMARK" == "0" ]]; then
  echo "RUN_BENCHMARK=0: TRT-LLM server running; waiting for shutdown."
  wait "$SERVER_PID"
  exit 0
fi

NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-10}"
if [[ -z "${NUM_PROMPTS:-}" ]]; then
  NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULTIPLIER))
fi

set +e
run_benchmark_serving \
  --model "$MODEL" \
  --port "$PORT" \
  --backend openai \
  --input-len "$ISL" \
  --output-len "$OSL" \
  --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONC" \
  --result-filename "$RESULT_FILENAME" \
  --result-dir /workspace/output/
BENCH_RC=$?
set -e

if [[ "$SCRAPE_METRICS" == "1" ]]; then
  fetch_perf_metrics || true
fi

if [[ "$NSYS_PROFILE" == "1" ]]; then
  if [[ "$NSYS_DURATION_SEC" -gt 0 ]]; then
    if [[ -n "${NSYS_ASYNC_PID:-}" ]]; then
      nsys_log "Waiting for async capture to complete..."
      wait "$NSYS_ASYNC_PID" || true
    fi
  else
    nsys_log ">>> CAPTURE STOP <<<"
    nsys stop --session="$NSYS_SESSION"
    finalize_nsys_report
  fi

  nsys_log "Shutting down session..."
  nsys shutdown --session="$NSYS_SESSION" || true
  nsys_log "Done."
else
  kill "$SERVER_PID" >/dev/null 2>&1 || true
fi

wait "$SERVER_PID" >/dev/null 2>&1 || true

exit "$BENCH_RC"
