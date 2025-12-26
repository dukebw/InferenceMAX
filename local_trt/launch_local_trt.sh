#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-nvcr.io#nvidia/tensorrt-llm/release:1.2.0rc5}"
MODEL="${MODEL:-nvidia/DeepSeek-R1-0528-FP4-V2}"
TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
CONC="${CONC:-4}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-160000}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
PORT="${PORT:-8888}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
RESULT_FILENAME="${RESULT_FILENAME:-dsr1_1k1k_fp4_trt_tp${TP}_ep${EP_SIZE}_dpa_${DP_ATTENTION}_conc${CONC}_local}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-}"

NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_SESSION="${NSYS_SESSION:-trtllm-profile}"
NSYS_OUT="${NSYS_OUT:-/workspace/output/${RESULT_FILENAME}}"
NSYS_EXPORT_SQLITE="${NSYS_EXPORT_SQLITE:-0}"
NSYS_START_BEFORE_HEALTH="${NSYS_START_BEFORE_HEALTH:-0}"
NSYS_DELAY_SEC="${NSYS_DELAY_SEC:-0}"
NSYS_DURATION_SEC="${NSYS_DURATION_SEC:-0}"
MTP_ENABLE="${MTP_ENABLE:-1}"
MTP_NUM_NEXTN_PREDICT_LAYERS="${MTP_NUM_NEXTN_PREDICT_LAYERS:-1}"
SCRAPE_METRICS="${SCRAPE_METRICS:-1}"
MODULAR_REPO="${MODULAR_REPO:-$HOME/work/modular}"
BENCH_BACKEND="${BENCH_BACKEND:-trtllm-chat}"
BENCH_HOST="${BENCH_HOST:-localhost}"
BENCH_PORT="${BENCH_PORT:-$PORT}"
BENCH_DATASET_NAME="${BENCH_DATASET_NAME:-random}"
BENCH_MODEL="${BENCH_MODEL:-$MODEL}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-300}"
BENCH_RANDOM_INPUT_LEN="${BENCH_RANDOM_INPUT_LEN:-75000}"
BENCH_RANDOM_OUTPUT_LEN="${BENCH_RANDOM_OUTPUT_LEN:-300}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-8}"
BENCH_RANDOM_SYS_PROMPT_RATIO="${BENCH_RANDOM_SYS_PROMPT_RATIO:-0.96}"
BENCH_SKIP_TEST_PROMPT="${BENCH_SKIP_TEST_PROMPT:-1}"
CHECK_PREFIX_CACHING="${CHECK_PREFIX_CACHING:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-trtllm-local}"
KEEP_CONTAINER="${KEEP_CONTAINER:-1}"
SKIP_SERVER="${SKIP_SERVER:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

IMAGE_RESOLVED="${IMAGE//#/\/}"

set -x

RUNTIME_ARGS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  RUNTIME_ARGS=(--runtime nvidia)
fi

HOST_LIB_MOUNTS=()
HOST_LIB_ENV=()
if [[ ${#RUNTIME_ARGS[@]} -eq 0 ]]; then
  HOST_LIB_DIR="/opt/host-libs"
  HOST_LIBCUDA=""
  HOST_LIBNVIDIA_ML=""
  HOST_LIBPTX=""

  if [[ -r /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
    HOST_LIBCUDA="/usr/lib/x86_64-linux-gnu/libcuda.so.1"
  elif [[ -r /usr/local/cuda-13.1/compat/libcuda.so.1 ]]; then
    HOST_LIBCUDA="/usr/local/cuda-13.1/compat/libcuda.so.1"
  fi

  if [[ -r /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ]]; then
    HOST_LIBNVIDIA_ML="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  elif [[ -r /usr/local/cuda-13.1/compat/libnvidia-ml.so.1 ]]; then
    HOST_LIBNVIDIA_ML="/usr/local/cuda-13.1/compat/libnvidia-ml.so.1"
  fi

  if [[ -r /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 ]]; then
    HOST_LIBPTX="/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"
  elif [[ -r /usr/local/cuda-13.1/compat/libnvidia-ptxjitcompiler.so.1 ]]; then
    HOST_LIBPTX="/usr/local/cuda-13.1/compat/libnvidia-ptxjitcompiler.so.1"
  fi

  if [[ -n "$HOST_LIBCUDA" ]]; then
    HOST_LIB_MOUNTS+=( -v "$HOST_LIBCUDA:$HOST_LIB_DIR/libcuda.so.1:ro" )
  fi
  if [[ -n "$HOST_LIBNVIDIA_ML" ]]; then
    HOST_LIB_MOUNTS+=( -v "$HOST_LIBNVIDIA_ML:$HOST_LIB_DIR/libnvidia-ml.so.1:ro" )
  fi
  if [[ -n "$HOST_LIBPTX" ]]; then
    HOST_LIB_MOUNTS+=( -v "$HOST_LIBPTX:$HOST_LIB_DIR/libnvidia-ptxjitcompiler.so.1:ro" )
  fi

  if [[ ${#HOST_LIB_MOUNTS[@]} -gt 0 ]]; then
    HOST_LIB_ENV=( -e "LD_LIBRARY_PATH=${HOST_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" )
  fi
fi

if [[ ! -x "${MODULAR_REPO}/bazelw" ]]; then
  echo "ERROR: bazelw not found at ${MODULAR_REPO}/bazelw" >&2
  exit 1
fi

if [[ "$SKIP_SERVER" == "1" ]]; then
  echo "INFO: SKIP_SERVER=1, skipping container startup. Assuming server is already running."
  # Simple health check without container monitoring
  if ! curl --silent --fail "http://${BENCH_HOST}:${BENCH_PORT}/health" >/dev/null; then
    echo "ERROR: Server not healthy at http://${BENCH_HOST}:${BENCH_PORT}/health" >&2
    exit 1
  fi
  echo "INFO: Server is healthy."
else
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi

  CONTAINER_ID="$(docker run --init --network host --detach \
    --name "$CONTAINER_NAME" \
    "${RUNTIME_ARGS[@]}" --gpus all --ipc host --privileged \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${REPO_ROOT}:/workspace" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${OUTPUT_DIR}:/workspace/output" \
    "${HOST_LIB_MOUNTS[@]}" \
    "${HOST_LIB_ENV[@]}" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HF_HUB_CACHE=/root/.cache/huggingface \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e NCCL_GRAPH_REGISTER=0 \
    -e MODEL="$MODEL" \
    -e TP="$TP" \
    -e EP_SIZE="$EP_SIZE" \
    -e DP_ATTENTION="$DP_ATTENTION" \
    -e CONC="$CONC" \
    -e ISL="$ISL" \
    -e BENCH_RANDOM_INPUT_LEN="$BENCH_RANDOM_INPUT_LEN" \
    -e OSL="$OSL" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO" \
    -e PORT="$PORT" \
    -e RESULT_FILENAME="$RESULT_FILENAME" \
    -e NUM_PROMPTS="$NUM_PROMPTS" \
    -e NUM_PROMPTS_MULTIPLIER="$NUM_PROMPTS_MULTIPLIER" \
    -e NSYS_PROFILE="$NSYS_PROFILE" \
    -e NSYS_TRACE="$NSYS_TRACE" \
    -e NSYS_CUDA_GRAPH_TRACE="$NSYS_CUDA_GRAPH_TRACE" \
    -e NSYS_SESSION="$NSYS_SESSION" \
    -e NSYS_OUT="$NSYS_OUT" \
    -e NSYS_EXPORT_SQLITE="$NSYS_EXPORT_SQLITE" \
    -e NSYS_START_BEFORE_HEALTH="$NSYS_START_BEFORE_HEALTH" \
    -e NSYS_DELAY_SEC="$NSYS_DELAY_SEC" \
    -e NSYS_DURATION_SEC="$NSYS_DURATION_SEC" \
    -e MTP_ENABLE="$MTP_ENABLE" \
    -e MTP_NUM_NEXTN_PREDICT_LAYERS="$MTP_NUM_NEXTN_PREDICT_LAYERS" \
    -e SCRAPE_METRICS="$SCRAPE_METRICS" \
    -e RUN_BENCHMARK=0 \
    --entrypoint=/bin/bash \
    "${IMAGE_RESOLVED}" \
    /workspace/local_trt/run_local_trt.sh)"

  cleanup() {
    if [[ "$KEEP_CONTAINER" == "1" ]]; then
      echo "INFO: Container $CONTAINER_NAME ($CONTAINER_ID) left running (KEEP_CONTAINER=1)."
    else
      docker rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  until curl --silent --fail "http://${BENCH_HOST}:${BENCH_PORT}/health" >/dev/null; do
    container_state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_ID" 2>/dev/null || true)"
    if [[ "$container_state" == "exited" || "$container_state" == "dead" || -z "$container_state" ]]; then
      docker logs "$CONTAINER_ID" 2>/dev/null || true
      echo "ERROR: TRT-LLM container exited before becoming healthy." >&2
      exit 1
    fi
    sleep 10
  done
fi

if [[ "$CHECK_PREFIX_CACHING" == "1" && "$BENCH_BACKEND" == modular* ]]; then
  if ! curl --silent --fail -X POST \
    "http://${BENCH_HOST}:${BENCH_PORT}/reset_prefix_cache" >/dev/null; then
    echo "ERROR: Prefix caching is not enabled for ${BENCH_MODEL}." >&2
    exit 1
  fi
fi

set -x
cd "${MODULAR_REPO}"
./bazelw run --config=disable-mypy \
  //max/python/max/benchmark:benchmark_serving -- \
  --backend "${BENCH_BACKEND}" \
  --host "${BENCH_HOST}" \
  --port "${BENCH_PORT}" \
  --dataset-name "${BENCH_DATASET_NAME}" \
  --model "${BENCH_MODEL}" \
  --request-rate "${BENCH_REQUEST_RATE}" \
  --num-prompts "${BENCH_NUM_PROMPTS}" \
  --random-input-len "${BENCH_RANDOM_INPUT_LEN}" \
  --random-output-len "${BENCH_RANDOM_OUTPUT_LEN}" \
  --max-concurrency "${BENCH_MAX_CONCURRENCY}" \
  --random-sys-prompt-ratio "${BENCH_RANDOM_SYS_PROMPT_RATIO}" \
  --collect-gpu-stats
