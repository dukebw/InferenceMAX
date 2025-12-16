#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

# Set up environment variables for SLURM
export SLURM_ACCOUNT="amd"
export SLURM_PARTITION="compute"
export SLURM_JOB_NAME="benchmark-sglang-standalone.job"

export SGL_SLURM_JOBS_PATH="sglang_disagg"

export MODEL_NAME="DeepSeek-R1"
# export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
# export MODEL_PATH="/apps/data/models"
export MODEL_PATH="/nfsdata"
export ISL="$ISL"
export OSL="$OSL"

bash benchmarks/"${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}_slurm.sh"

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
    echo "Jobs still running..."
    squeue --steps -u $USER
    sleep 30
done

# FIXME: The below is bad and is a result of the indirection of the ways in which
# Dynamo jobs are launched. In a follow-up PR, the location of the result file should not
# depend on the runner, it should always be in the same spot in the GH workspace.

# Process results from all configurations

# TODO(billishyahao): process the log file...
# search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
# Find the latest log directory that contains the data

sudo chown -R $USER:$(id -gn) $SGL_SLURM_JOBS_PATH/logs

cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

LOGS_DIR=$(python3 collect_latest_results.py "$SGL_SLURM_JOBS_PATH" $ISL $OSL 1)
if [ -z "$LOGS_DIR" ]; then
    echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"
ls -la $LOGS_DIR

# Result JSON are contained within the result directory
for result_file in $(find $LOGS_DIR -type f); do
    # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
    file_name=$(basename $result_file)
    if [ -f $result_file ]; then
        # Copy the result file to workspace with a unique name
        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
        echo "Found result file ${result_file}. Copying them to ${WORKSPACE_RESULT_FILE}"
        cp $result_file $WORKSPACE_RESULT_FILE
    fi
done

echo "All result files processed"
