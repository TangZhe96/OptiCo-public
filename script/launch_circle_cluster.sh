#!/bin/bash
# Start OptiCo-public run_circle on every hostfile node (not master-only).
# Caller must already export OPTICO_ROOT_PATH, OUTPUT_DIR, API keys, model env.
set -euo pipefail

ROOT=${OPTICO_ROOT_PATH:?OPTICO_ROOT_PATH required}
HOSTFILE=${MPI_HOSTFILE:-/etc/mpi/hostfile}
SSH_PORT=${PLANNER_SSH_PORT:-8081}
CIRCLE="${ROOT}/script/run_circle.sh"

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
N=${#HOSTS[@]}
if [ "${N}" -lt 1 ]; then
  echo "ERROR: empty hostfile ${HOSTFILE}" >&2
  exit 1
fi

export NUM_NODES=${NUM_NODES:-${N}}
export WORLD_SIZE=${NUM_NODES}
export MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]}}
export MASTER_PORT=${MASTER_PORT:-29500}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

echo "CLUSTER n=${N} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-unset} MODEL_TYPE=${MODEL_TYPE:-unset} MODEL_SIZE=${MODEL_SIZE:-unset}"

pass_keys=(
  OPTICO_ROOT_PATH META_OPTICO_ROOT NHMEGATRON_PATH
  OUTPUT_DIR OUTPUT_PATH
  API_KEY KEY BASE_URL INFER_MODEL REASONING_EFFORT LLM_API_MODE
  WORLD_SIZE NUM_NODES MASTER_ADDR MASTER_PORT
  CUDA_VISIBLE_DEVICES GPUS_PER_NODE TQ_GPU_NUM
  NUMBER_OF_CYCLES TRAIN_ITERS
  MODEL MODEL_TYPE MODEL_SIZE PLATFORM
  DATASET_PATH TOKENIZER_PATH PRETRAIN_CHECKPOINT_PATH
  PR GLOBAL_BATCH_SIZE SEQ_LEN SEQ_LENGTH
  GPU_SMI GPU_MEMORY_LIMIT_MB GPU_PEAK_TFLOPS
  NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
)

env_exports=""
for k in "${pass_keys[@]}"; do
  if [ -n "${!k+x}" ] && [ -n "${!k}" ]; then
    env_exports+="export ${k}=$(printf '%q' "${!k}"); "
  fi
done

LAUNCH_DIR="${OUTPUT_DIR:-${ROOT}/output}/_cluster_launch"
mkdir -p "${LAUNCH_DIR}"

CIRCLE_ARGS=""
if [ -n "${MODEL_TYPE:-}" ]; then
  CIRCLE_ARGS="${MODEL_TYPE} ${MODEL_SIZE:-A2.4B} ${NUMBER_OF_CYCLES:-20}"
fi

launch_rank() {
  local rank="$1"
  local host="$2"
  local log="${LAUNCH_DIR}/rank${rank}.ssh.log"
  local inner
  inner="${env_exports} export RANK=${rank}; export POD_RANK=${rank}; export WORLD_SIZE=${N}; export NUM_NODES=${N}; "
  inner+="if [ -f /root/miniconda/etc/profile.d/conda.sh ]; then source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda; fi; "
  inner+="if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null; then mkdir -p /tmp/optico-pybin; ln -sfn \$(command -v python3) /tmp/optico-pybin/python; export PATH=/tmp/optico-pybin:\$PATH; fi; "
  inner+="cd $(printf '%q' "${ROOT}"); umask 000; exec bash $(printf '%q' "${CIRCLE}") ${CIRCLE_ARGS}"
  nohup ssh -n -p "${SSH_PORT}" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=15 \
    "${host}" \
    "bash --noprofile --norc -c $(printf '%q' "${inner}")" \
    >"${log}" 2>&1 &
  echo "launched rank=${rank} host=${host} ssh_pid=$! log=${log}"
}

for i in "${!HOSTS[@]}"; do
  launch_rank "${i}" "${HOSTS[$i]}"
  sleep 0.15
done

echo "verify ${N} ranks are alive (poll up to 120s)..."
alive=0
for _try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  alive=0
  missing=""
  for host in "${HOSTS[@]}"; do
    if timeout 10 ssh -n -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=6 \
      "${host}" 'pgrep -f run_circle.sh >/dev/null'; then
      alive=$((alive + 1))
    else
      missing="${missing} ${host}"
    fi
  done
  echo "try=${_try} alive_ranks=${alive}/${N}"
  if [ "${alive}" -eq "${N}" ]; then
    break
  fi
  sleep 10
done
if [ "${alive}" -lt "${N}" ]; then
  echo "ERROR: only ${alive}/${N} nodes started run_circle; missing:${missing}" >&2
  echo "rank ssh logs under ${LAUNCH_DIR}" >&2
  exit 1
fi

echo "waiting for run_circle ranks to exit"
dead_rounds=0
while true; do
  alive_now=0
  for host in "${HOSTS[@]}"; do
    if timeout 10 ssh -n -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=6 \
      "${host}" 'pgrep -f run_circle.sh >/dev/null'; then
      alive_now=$((alive_now + 1))
    fi
  done
  echo "alive_ranks=${alive_now}/${N}"
  if [ "${alive_now}" -eq 0 ]; then
    dead_rounds=$((dead_rounds + 1))
    echo "no run_circle ranks alive (${dead_rounds}/3)"
    if [ "${dead_rounds}" -ge 3 ]; then
      echo "cluster run_circle finished"
      break
    fi
  else
    dead_rounds=0
  fi
  sleep 15
done
