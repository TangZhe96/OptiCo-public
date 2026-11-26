#!/bin/bash
set -eo pipefail

# OptiCo-public → Meta-OptiCo nhmegatron (W64 021 MoE).
# Round-1 A2.4B: weak OptiCo search start (TP=1 PP=1 EP=32 AC=full), not vendor
#   pretrain.sh EP=8 AC=none. Monitor + this script must stay aligned.
# 32B/236B/MINI empty knobs: USE_SCRIPT_DEFAULT_CONFIG=true → pretrain.sh.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPTICO_ROOT_PATH=${OPTICO_ROOT_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export META_OPTICO_ROOT=${META_OPTICO_ROOT:-$(cd "${OPTICO_ROOT_PATH}/../Meta-OptiCo" && pwd)}
export NHMEGATRON_PATH=${NHMEGATRON_PATH:-"${META_OPTICO_ROOT}/nhmegatron"}

export MODEL=${MODEL:-021}
export MODEL_TYPE=${MODEL_TYPE:-021}
export PLATFORM=${PLATFORM:-W64}

# Normalize size aliases (run_circle must not leave A21B for 021).
case "${MODEL_SIZE}" in
    32b|32B) export MODEL_SIZE=32B ;;
    236b|236B|A21B|A21b|a21b|A236B) export MODEL_SIZE=236B ;;
    mini|MINI) export MODEL_SIZE=MINI ;;
esac
export MODEL_SIZE=${MODEL_SIZE:-32B}

map_pp_layers() {
    local v="${1:-none}"
    local l
    l=$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')
    case "${l}" in
        ""|none|null) echo 0 ;;
        *) echo "${v}" ;;
    esac
}

apply_pretrain_defaults() {
    # Size-specific fills. A2.4B = weak OptiCo start; others ≈ pretrain.sh.
    export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
    export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-4096}}
    export PR=${PR:-bf16}
    export CP=${CP:-1}
    export DO=${DO:-true}
    export SP=${SP:-false}
    export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
    export FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY:-false}
    export TRAIN_ITERS=${TRAIN_ITERS:-5}
    export SAVE_CKPT=${SAVE_CKPT:-false}
    export DIST_BACKEND=${DIST_BACKEND:-nccl}
    export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
    export AC_METHOD=${AC_METHOD:-uniform}
    export AC_MODULES=${AC_MODULES:-none}
    case "${MODEL_SIZE}" in
        A2.4b|16B|16b|A16B) export MODEL_SIZE=A2.4B ;;
    esac
    if [ "${MODEL_SIZE}" = "236B" ]; then
        export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
        export TP=${TP:-1}
        export PP=${PP:-4}
        export EP=${EP:-32}
        export AC=${AC:-full}
        export AC_LAYERS=${AC_LAYERS:-1}
        export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
        export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
        export LR=${LR:-2.4e-4}
        export MIN_LR=${MIN_LR:-2.4e-5}
    elif [ "${MODEL_SIZE}" = "MINI" ]; then
        export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
        export TP=${TP:-1}
        export PP=${PP:-1}
        export EP=${EP:-1}
        export AC=${AC:-full}
        export AC_LAYERS=${AC_LAYERS:-1}
        export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
        export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
        export LR=${LR:-3.2e-4}
        export MIN_LR=${MIN_LR:-3.2e-5}
    elif [ "${MODEL_SIZE}" = "A2.4B" ]; then
        # Weak search start (EP=1 OOMs; EP=32 divides 64 experts, world%EP==0).
        export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
        export TP=${TP:-1}
        export PP=${PP:-1}
        export EP=${EP:-32}
        export AC=${AC:-full}
        export AC_METHOD=${AC_METHOD:-uniform}
        export AC_LAYERS=${AC_LAYERS:-1}
        export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
        export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
        export LR=${LR:-3.2e-4}
        export MIN_LR=${MIN_LR:-3.2e-5}
    else
        # 32B
        export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
        export TP=${TP:-1}
        export PP=${PP:-4}
        export EP=${EP:-4}
        export AC=${AC:-full}
        export AC_LAYERS=${AC_LAYERS:-1}
        export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
        export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
        export LR=${LR:-3.2e-4}
        export MIN_LR=${MIN_LR:-3.2e-5}
    fi
    export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS:-1}}
}

export DATASET_PATH=${DATASET_PATH:-"/mnt/hw-nas/dataset/021-W64/code2_text_document"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/hw-nas/model/021-W64/tokenizer"}
export PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"none"}
export TASK=${TASK:-pretrain}
export MFU_BASE_VALUE=${MFU_BASE_VALUE:-${GPU_PEAK_TFLOPS:-280}}

case "${MODEL_SIZE}" in
    A2.4b|16B|16b|A16B) export MODEL_SIZE=A2.4B ;;
esac

if [ -z "${TP:-}" ] && [ -z "${PP:-}" ] && [ -z "${EP:-}" ] && [ -z "${AC:-}" ]; then
    if [ "${MODEL_SIZE}" = "A2.4B" ]; then
        # OptiCo circle: weak start (do not fall through to vendor EP=8 AC=none).
        apply_pretrain_defaults
        case "${AC,,}" in
            full)
                case "${AC_METHOD,,}" in
                    uniform|block) export AC_METHOD="${AC_METHOD,,}" ;;
                    *) export AC_METHOD=uniform ;;
                esac
                export AC_MODULES=${AC_MODULES:-none}
                ;;
        esac
        export MP_PP0_LAYERS=${MP_PP0_LAYERS:-$(map_pp_layers "${FIRST_PP_NUM_LAYERS:-none}")}
        export MP_PPN_LAYERS=${MP_PPN_LAYERS:-$(map_pp_layers "${LAST_PP_NUM_LAYERS:-none}")}
        export USE_SCRIPT_DEFAULT_CONFIG=false
        echo "OptiCo-public 021/W64 A2.4B: weak start TP=${TP} PP=${PP} EP=${EP} AC=${AC}"
    else
        # Direct launch / no monitor presets: real script defaults.
        unset TP PP CP EP AC MP_AC_LAYERS AC_LAYERS AC_METHOD AC_MODULES RECOMPUTE_METHOD
        unset MP_PP0_LAYERS MP_PPN_LAYERS FIRST_PP_NUM_LAYERS LAST_PP_NUM_LAYERS
        unset SP DO OVERLAP_GRAD_REDUCE OVERLAP_PARAM_GATHER TP_COMM_OVERLAP
        unset FUSED_CROSS_ENTROPY TRAIN_ITERS EXIT_INTERVAL LR MIN_LR
        unset OPTIMIZER_OFFLOAD MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE SEQ_LEN SEQ_LENGTH
        unset PR DIST_BACKEND SAVE_CKPT
        export USE_SCRIPT_DEFAULT_CONFIG=true
        echo "OptiCo-public 021/W64: USE_SCRIPT_DEFAULT_CONFIG=true (W64/021/pretrain.sh defaults)"
    fi
else
    # PARA or monitor round-1 presets present.
    # Empty-_LAUNCH poison looks like TP=1 PP=1 EP=1 AC=none — drop and refill
    # with size defaults (A2.4B → weak full AC; others → pretrain-like).
    if [ "${AC,,}" = "none" ] && [ "${TP:-}" = "1" ] && [ "${PP:-}" = "1" ] && [ "${EP:-}" = "1" ]; then
        echo "warn: dropping fail-report poison TP=1 PP=1 EP=1 AC=none; applying size defaults"
        unset TP PP EP AC AC_METHOD AC_LAYERS AC_MODULES MP_AC_LAYERS
        unset SP OVERLAP_GRAD_REDUCE OVERLAP_PARAM_GATHER TP_COMM_OVERLAP FUSED_CROSS_ENTROPY
    fi
    apply_pretrain_defaults
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
    esac
    export MP_PP0_LAYERS=${MP_PP0_LAYERS:-$(map_pp_layers "${FIRST_PP_NUM_LAYERS:-none}")}
    export MP_PPN_LAYERS=${MP_PPN_LAYERS:-$(map_pp_layers "${LAST_PP_NUM_LAYERS:-none}")}
    export USE_SCRIPT_DEFAULT_CONFIG=false
fi

TRAIN_SH="${META_OPTICO_ROOT}/script/train/train.sh"
if [ ! -f "${TRAIN_SH}" ]; then
    echo "Error: Meta-OptiCo train.sh not found: ${TRAIN_SH}"
    echo "Set META_OPTICO_ROOT / NHMEGATRON_PATH (nhmegatron under Meta-OptiCo)."
    exit 1
fi

echo "OptiCo-public 021/W64 via ${TRAIN_SH}"
echo "  MODEL_SIZE=${MODEL_SIZE} PLATFORM=${PLATFORM} NHMEGATRON_PATH=${NHMEGATRON_PATH}"
echo "  USE_SCRIPT_DEFAULT_CONFIG=${USE_SCRIPT_DEFAULT_CONFIG} TP=${TP:-<script>} PP=${PP:-<script>} EP=${EP:-<script>} AC=${AC:-<script>} GBS=${GLOBAL_BATCH_SIZE:-<script>}"
bash "${TRAIN_SH}"
