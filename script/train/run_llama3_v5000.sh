#!/bin/bash
set -eo pipefail

# OptiCo-public → Meta-OptiCo nhmegatron (V5000 Llama3).
# Keeps public parameter names (AC_LAYERS / FIRST_PP_NUM_LAYERS / ...) and
# maps them onto nhmegatron knobs before calling Meta-OptiCo train.sh.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPTICO_ROOT_PATH=${OPTICO_ROOT_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export META_OPTICO_ROOT=${META_OPTICO_ROOT:-$(cd "${OPTICO_ROOT_PATH}/../Meta-OptiCo" && pwd)}
export NHMEGATRON_PATH=${NHMEGATRON_PATH:-"${META_OPTICO_ROOT}/nhmegatron"}

export MODEL=${MODEL:-llama3}
export MODEL_TYPE=${MODEL_TYPE:-llama3}
export PLATFORM=${PLATFORM:-V5000}
export MODEL_SIZE=${MODEL_SIZE:-70B}

# Same defaults as nhmegatron zj_examples/V5000/llama3/pretrain.sh.
# PARA_FILE / env from run_monitor.sh win when already set.
if [ "${MODEL_SIZE}" = "70B" ]; then
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
    export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
    export TP=${TP:-8}
    export PP=${PP:-8}
    export AC=${AC:-full}
    export AC_LAYERS=${AC_LAYERS:-1}
    export TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/hw-nas/model/Meta-Llama-3-70B"}
elif [ "${MODEL_SIZE}" = "8B" ]; then
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
    export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
    export TP=${TP:-1}
    export PP=${PP:-1}
    export AC=${AC:-full}
    export AC_LAYERS=${AC_LAYERS:-6}
    export TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/hw-nas/model/Meta-Llama-3-8B"}
elif [ "${MODEL_SIZE}" = "0.5B" ]; then
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
    export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
    export TP=${TP:-1}
    export PP=${PP:-1}
    export AC=${AC:-full}
    export AC_LAYERS=${AC_LAYERS:-6}
    export TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/hw-nas/model/Meta-Llama-3-8B"}
else
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE} (use 0.5B, 8B or 70B)"
    exit 1
fi
# Never leave tokenizer as the literal "none" (common when PRETRAIN_CHECKPOINT_PATH=none).
if [ -z "${TOKENIZER_PATH}" ] || [ "${TOKENIZER_PATH}" = "none" ]; then
    export TOKENIZER_PATH="/mnt/hw-nas/model/Meta-Llama-3-70B"
fi
export PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-none}

map_pp_layers() {
    local v="${1:-none}"
    local l
    l=$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')
    case "${l}" in
        ""|none|null) echo 0 ;;
        *) echo "${v}" ;;
    esac
}

export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS:-1}}
# Megatron only accepts --recompute-method {uniform,block}, and only with AC=full.
case "${AC,,}" in
    full)
        case "${AC_METHOD,,}" in
            uniform|block) export AC_METHOD="${AC_METHOD,,}" ;;
            *) export AC_METHOD=uniform ;;
        esac
        export AC_MODULES=${AC_MODULES:-none}
        ;;
    sel|selective)
        export AC=sel
        export AC_METHOD=${AC_METHOD:-none}
        if [ -z "${AC_MODULES:-}" ] || [ "${AC_MODULES,,}" = "none" ]; then
            export AC_MODULES=core_attn
        fi
        ;;
    *)
        export AC_METHOD=${AC_METHOD:-none}
        export AC_MODULES=${AC_MODULES:-none}
        ;;
esac
export MP_PP0_LAYERS=${MP_PP0_LAYERS:-$(map_pp_layers "${FIRST_PP_NUM_LAYERS:-none}")}
export MP_PPN_LAYERS=${MP_PPN_LAYERS:-$(map_pp_layers "${LAST_PP_NUM_LAYERS:-none}")}
export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
export PR=${PR:-bf16}
[ -z "${MICRO_BATCH_SIZE}" ] && MICRO_BATCH_SIZE=1
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export CP=${CP:-1}
export EP=${EP:-1}
export DO=${DO:-true}
export SP=${SP:-true}
export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
export FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY:-false}
export TRAIN_ITERS=${TRAIN_ITERS:-5}
export LR=${LR:-5e-5}
export MIN_LR=${MIN_LR:-5e-6}
export SAVE_CKPT=${SAVE_CKPT:-false}
export DIST_BACKEND=${DIST_BACKEND:-nccl}
export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
export USE_SCRIPT_DEFAULT_CONFIG=false
export TASK=${TASK:-pretrain}
export MFU_BASE_VALUE=${MFU_BASE_VALUE:-${GPU_PEAK_TFLOPS:-312}}

TRAIN_SH="${META_OPTICO_ROOT}/script/train/train.sh"
if [ ! -f "${TRAIN_SH}" ]; then
    echo "Error: Meta-OptiCo train.sh not found: ${TRAIN_SH}"
    echo "Set META_OPTICO_ROOT / NHMEGATRON_PATH (nhmegatron under Meta-OptiCo)."
    exit 1
fi

echo "OptiCo-public llama3 via ${TRAIN_SH}"
echo "  MODEL_SIZE=${MODEL_SIZE} PLATFORM=${PLATFORM} NHMEGATRON_PATH=${NHMEGATRON_PATH}"
echo "  TP=${TP} PP=${PP} CP=${CP} AC=${AC} AC_METHOD=${AC_METHOD} MBS=${MICRO_BATCH_SIZE} GBS=${GLOBAL_BATCH_SIZE}"
bash "${TRAIN_SH}"
