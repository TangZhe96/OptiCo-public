#!/bin/bash
set +e

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning,ignore::ResourceWarning,ignore::DeprecationWarning"
export NCCL_WARN=ERROR
export NCCL_TIMEOUT=300

MODEL_TYPE=${MODEL_TYPE:-${MODEL:-"qwen2.5"}}
MODEL=${MODEL:-${MODEL_TYPE}}
MODEL_SIZE=${MODEL_SIZE:-"0.5B"}

MONITOR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH=$(cd "${MONITOR_DIR}/.." && pwd)
export OPTICO_ROOT_PATH=${OPTICO_ROOT_PATH:-$(cd "${SCRIPT_PATH}/.." && pwd)}

OUTPUT_PATH=${OUTPUT_PATH:-"${SCRIPT_PATH}/../output_${MODEL_TYPE}_${MODEL_SIZE}"}
REPORT_FILE=${REPORT_FILE:-"${OUTPUT_PATH}/report.csv"}
PARA_FILE=${PARA_FILE:-"${OUTPUT_PATH}/parameters.txt"}
FIXED_PARA_FILE=${FIXED_PARA_FILE:-"${OUTPUT_PATH}/fixed_parameters.txt"}
RESULT_FILE=${RESULT_FILE:-"${OUTPUT_PATH}/monitor_result.txt"}
ERROR_LOG_FILE=${ERROR_LOG_FILE:-"${OUTPUT_PATH}/monitor_error.txt"}

ITER_CUR=${ITER_CUR:-"MONITOR_TEST"}
CURRENT_LOG_DIR=${OUTPUT_PATH}/${ITER_CUR}
mkdir -p ${CURRENT_LOG_DIR}
MEGATRON_LOG_DIR="${CURRENT_LOG_DIR}/megatron_log"
mkdir -p ${MEGATRON_LOG_DIR}
export LOG_FILE="${MEGATRON_LOG_DIR}/${RANK}.txt"
export MEMORY_LOG_FILE="${CURRENT_LOG_DIR}/memory_rank${RANK}.txt"

is_moe_model() {
    local t="${MODEL_TYPE,,}"
    case "${t}" in
        *moe*|deepseek*|deepseek_v2|021) return 0 ;;
        *) return 1 ;;
    esac
}

map_pp_layers() {
    local v="${1:-none}"
    local l
    l=$(printf '%s' "${v}" | tr '[:upper:]' '[:lower:]')
    case "${l}" in
        ""|none|null) echo 0 ;;
        *) echo "${v}" ;;
    esac
}

_SAVED_WORLD_SIZE="${WORLD_SIZE:-}"
_SAVED_NUM_NODES="${NUM_NODES:-}"
_SAVED_RANK="${RANK:-}"

if [ -f "${PARA_FILE}" ]; then
    echo "read parameters"
    while IFS=':' read -r key value || [ -n "$key" ]; do
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        case "${key}" in
            NUM_NODES|WORLD_SIZE|RANK|POD_RANK) continue ;;
        esac
        # Never import blank values: a failed prior cycle can write
        # "MICRO_BATCH_SIZE: " into PARA_FILE and then train_LLM rejects it.
        if [ -n "$key" ] && [ -n "$value" ]; then
            export "$key=$value"
            echo "export $key=$value"
        else
            echo "skip empty para: ${key}"
        fi
    done < "${PARA_FILE}"
fi

if [ -n "${_SAVED_NUM_NODES}" ]; then
    export NUM_NODES="${_SAVED_NUM_NODES}"
fi
if [ -n "${_SAVED_WORLD_SIZE}" ]; then
    export WORLD_SIZE="${_SAVED_WORLD_SIZE}"
fi
if [ -n "${_SAVED_RANK}" ]; then
    export RANK="${_SAVED_RANK}"
fi
# Re-bind LOG_FILE after RANK restore (PARA must not clobber it).
export LOG_FILE="${MEGATRON_LOG_DIR}/${RANK}.txt"
: > "${LOG_FILE}"

# OptiCo-public names → nhmegatron (ignored by Pai-Megatron scripts).
export MODEL="${MODEL:-${MODEL_TYPE}}"
# Only map OptiCo names when PARA already set them. Do NOT invent TP/EP/AC
# here — that would shadow run_deepseek_v2.sh model presets (${TP:-4} etc).
if [ -n "${AC_LAYERS}${MP_AC_LAYERS}" ]; then
    export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS}}
fi
if [ -n "${FIRST_PP_NUM_LAYERS}${MP_PP0_LAYERS}" ]; then
    export MP_PP0_LAYERS=${MP_PP0_LAYERS:-$(map_pp_layers "${FIRST_PP_NUM_LAYERS:-none}")}
fi
if [ -n "${LAST_PP_NUM_LAYERS}${MP_PPN_LAYERS}" ]; then
    export MP_PPN_LAYERS=${MP_PPN_LAYERS:-$(map_pp_layers "${LAST_PP_NUM_LAYERS:-none}")}
fi
export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-4096}}
# Soft knobs ok to default; parallelism/AC come from PARA or train script.
# 021 A2.4B round-1 uses weak OptiCo start (SP=false); do not force SP=true.
export CP=${CP:-1}
if [ "${MODEL_TYPE}" != "021" ] && [ "${MODEL}" != "021" ]; then
    export SP=${SP:-true}
fi
export DO=${DO:-true}
# Do not default AC_METHOD=none here — that poisons AC=full (Megatron rejects
# --recompute-method none). run_deepseek_v2.sh / pretrain.sh normalize it.
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
# Blank MBS from a poisoned PARA_FILE must not win.
[ -z "${MICRO_BATCH_SIZE}" ] && export MICRO_BATCH_SIZE=1
# If PARA did not set parallelism/AC, mirror train-script presets so
# round-1 (and fail-without-torchrun report rows) match that script.
if [ "${MODEL_TYPE}" = "deepseek_v2" ] || [ "${MODEL}" = "deepseek_v2" ]; then
    if [ -z "${TP}" ] || [ -z "${EP}" ] || [ -z "${AC}" ]; then
        case "${MODEL_SIZE}" in
            A2.4B|16B|16b|A16B)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
                export TP=${TP:-4}
                export PP=${PP:-1}
                export EP=${EP:-8}
                export AC=${AC:-full}
                export AC_LAYERS=${AC_LAYERS:-1}
                ;;
            A21B|236B|236b|A236B)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
                export TP=${TP:-8}
                export PP=${PP:-1}
                export EP=${EP:-32}
                export AC=${AC:-full}
                export AC_LAYERS=${AC_LAYERS:-20}
                ;;
        esac
        export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS}}
    fi
elif [ "${MODEL_TYPE}" = "021" ] || [ "${MODEL}" = "021" ]; then
    # Round-1 presets for report/_LAUNCH (must match run_021_w64.sh).
    # A2.4B: weak OptiCo search start (not vendor pretrain.sh EP=8 AC=none).
    # 32B/236B/MINI: still mirror W64/021/pretrain.sh.
    case "${MODEL_SIZE}" in
        A21B|A21b|a21b) export MODEL_SIZE=236B ;;
        16B|16b|A16B) export MODEL_SIZE=A2.4B ;;
    esac
    if [ -z "${TP}" ] || [ -z "${EP}" ] || [ -z "${AC}" ]; then
        case "${MODEL_SIZE}" in
            A2.4B|A2.4b)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
                export TP=${TP:-1}
                export PP=${PP:-1}
                export EP=${EP:-32}
                export AC=${AC:-full}
                export AC_METHOD=${AC_METHOD:-uniform}
                export AC_LAYERS=${AC_LAYERS:-1}
                export SP=${SP:-false}
                export DO=${DO:-true}
                export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
                export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
                export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-false}
                export FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY:-false}
                ;;
            32B|32b)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
                export TP=${TP:-1}
                export PP=${PP:-4}
                export EP=${EP:-4}
                export AC=${AC:-full}
                export AC_METHOD=${AC_METHOD:-uniform}
                export AC_LAYERS=${AC_LAYERS:-1}
                export SP=${SP:-false}
                export DO=${DO:-true}
                export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
                export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
                ;;
            236B|236b)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4096}
                export TP=${TP:-1}
                export PP=${PP:-4}
                export EP=${EP:-32}
                export AC=${AC:-full}
                export AC_METHOD=${AC_METHOD:-uniform}
                export AC_LAYERS=${AC_LAYERS:-1}
                export SP=${SP:-false}
                export DO=${DO:-true}
                export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-true}
                export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-true}
                ;;
            MINI|mini)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
                export TP=${TP:-1}
                export PP=${PP:-1}
                export EP=${EP:-1}
                export AC=${AC:-full}
                export AC_METHOD=${AC_METHOD:-uniform}
                export AC_LAYERS=${AC_LAYERS:-1}
                export SP=${SP:-false}
                ;;
        esac
        export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS}}
        export AC_MODULES=${AC_MODULES:-none}
    fi
elif { [ "${MODEL_TYPE}" = "llama3" ] || [ "${MODEL}" = "llama3" ]; } && [ "${PLATFORM}" = "V5000" ]; then
    if [ -z "${TP}" ] || [ -z "${PP}" ] || [ -z "${AC}" ]; then
        case "${MODEL_SIZE}" in
            70B|70b)
                # Keep in sync with script/train/run_llama3_v5000.sh
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
                export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
                export TP=${TP:-8}
                export PP=${PP:-8}
                export AC=${AC:-full}
                export AC_LAYERS=${AC_LAYERS:-1}
                ;;
            8B|8b)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
                export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
                export TP=${TP:-1}
                export PP=${PP:-1}
                export AC=${AC:-full}
                export AC_LAYERS=${AC_LAYERS:-6}
                ;;
            0.5B|0.5b)
                export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
                export SEQ_LEN=${SEQ_LEN:-${SEQ_LENGTH:-8192}}
                export TP=${TP:-1}
                export PP=${PP:-1}
                export AC=${AC:-full}
                export AC_LAYERS=${AC_LAYERS:-6}
                ;;
        esac
        export MP_AC_LAYERS=${MP_AC_LAYERS:-${AC_LAYERS}}
    fi
fi
# Round 1 has no PARA file. These match run_llama2.sh, so the cache key is the
# config that script will actually launch (PP=8, AC=full, uniform).
if [ "${MODEL_TYPE}" = "llama2" ] && [ "${MODEL_SIZE}" = "7B" ]; then
    export TP=${TP:-1}
    export PP=${PP:-8}
    export AC=${AC:-full}
    export AC_LAYERS=${AC_LAYERS:-1}
    export AC_MODULES=${AC_MODULES:-none}
    export OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
    export OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
    export FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY:-false}
    export FIRST_PP_NUM_LAYERS=${FIRST_PP_NUM_LAYERS:-none}
    export LAST_PP_NUM_LAYERS=${LAST_PP_NUM_LAYERS:-none}
    if [ "${AC}" = "full" ] && { [ -z "${AC_METHOD}" ] || [ "${AC_METHOD}" = "none" ]; }; then
        export AC_METHOD=uniform
    fi
fi
# Snapshot launch knobs before train; used if megatron_log has no torchrun line.
_LAUNCH_MBS="${MICRO_BATCH_SIZE}"
_LAUNCH_TP="${TP}"
_LAUNCH_PP="${PP}"
_LAUNCH_CP="${CP}"
_LAUNCH_EP="${EP}"
_LAUNCH_SP="${SP}"
_LAUNCH_DO="${DO}"
_LAUNCH_AC="${AC}"
_LAUNCH_AC_METHOD="${AC_METHOD}"
_LAUNCH_AC_LAYERS="${AC_LAYERS}"
_LAUNCH_AC_MODULES="${AC_MODULES}"
_LAUNCH_OGR="${OVERLAP_GRAD_REDUCE}"
_LAUNCH_OPG="${OVERLAP_PARAM_GATHER}"
_LAUNCH_TPCO="${TP_COMM_OVERLAP}"
_LAUNCH_FCE="${FUSED_CROSS_ENTROPY}"
_LAUNCH_PP0="${FIRST_PP_NUM_LAYERS}"
_LAUNCH_PPN="${LAST_PP_NUM_LAYERS}"
_LAUNCH_GBS="${GLOBAL_BATCH_SIZE}"

# Memory sampler on every node (PP last stage may not live on rank 0).
MONITOR_GPU_PID=""
MONITOR_GPU_SCRIPT="${SCRIPT_PATH}/monitor/monitor_gpu.sh"
echo "monitor memory rank=${RANK} (GPU_SMI=${GPU_SMI:-nvidia-smi}) -> ${MEMORY_LOG_FILE}"
if [ -f "${MONITOR_GPU_SCRIPT}" ]; then
    chmod +x "${MONITOR_GPU_SCRIPT}"
    nohup "${MONITOR_GPU_SCRIPT}" >/dev/null 2>&1 &
    MONITOR_GPU_PID=$!
fi

# Training
_suffix=""
if [ "${OPTICO_TRAIN_MULTINODE:-0}" = "1" ]; then
    _suffix="_multinode"
fi
if [ "${MODEL_TYPE}" = "deepseek_v2" ] || [ "${MODEL}" = "deepseek_v2" ]; then
    train_script=${SCRIPT_PATH}/train/run_deepseek_v2.sh
elif [ "${MODEL_TYPE}" = "021" ] || [ "${MODEL}" = "021" ]; then
    train_script=${SCRIPT_PATH}/train/run_021_w64.sh
elif { [ "${MODEL_TYPE}" = "llama3" ] || [ "${MODEL}" = "llama3" ]; } && [ "${PLATFORM}" = "V5000" ]; then
    train_script=${SCRIPT_PATH}/train/run_llama3_v5000.sh
elif [ "${MODEL_TYPE}" = "qwen2.5" ]; then
    train_script=${SCRIPT_PATH}/train/run_qwen${_suffix}.sh
elif [ "${MODEL_TYPE}" = "llama3" ]; then
    train_script=${SCRIPT_PATH}/train/run_llama3${_suffix}.sh
elif [ "${MODEL_TYPE}" = "llama2" ]; then
    train_script=${SCRIPT_PATH}/train/run_llama2${_suffix}.sh
elif [ "${MODEL_TYPE}" = "qwen3-moe" ]; then
    train_script=${SCRIPT_PATH}/train/run_qwen3_moe${_suffix}.sh
elif [ "${MODEL_TYPE}" = "gpt3" ]; then
    train_script=${SCRIPT_PATH}/train/run_gpt3${_suffix}.sh
else
    echo "ERROR: unknown MODEL_TYPE=${MODEL_TYPE}"
    train_script=""
fi
if [ -n "${train_script}" ]; then
    run_cmd="bash ${train_script}"
    echo "run cmd: ${run_cmd}" | tee -a "${LOG_FILE}"
    # Train scripts already `tee -a $LOG_FILE`. That tee also copies the same
    # stream to stdout, so piping stdout back into LOG_FILE writes every line
    # twice. `tail -n +3` then keeps the warmup iteration in the TFLOPS average.
    # Stdout goes to a side file; only lines the train script did not already
    # append (startup failures) are added to LOG_FILE.
    set +e
    # If one rank OOMs and exits, the other rank must abort too. Otherwise it
    # stays inside torchrun and misses every later round.
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120
    setsid bash "${train_script}" > "${CURRENT_LOG_DIR}/train_stdout.txt" 2>&1 &
    train_pid=$!
    while kill -0 "${train_pid}" 2>/dev/null; do
        if [ -f "${CURRENT_LOG_DIR}/peer_failed" ]; then
            kill -9 -"${train_pid}" 2>/dev/null || kill -9 "${train_pid}" 2>/dev/null
            break
        fi
        sleep 5
    done
    wait "${train_pid}"
    train_exit=$?
    if [ "${train_exit}" -ne 0 ]; then
        touch "${CURRENT_LOG_DIR}/peer_failed"
    fi
    set +e
    if [ -s "${CURRENT_LOG_DIR}/train_stdout.txt" ] && [ -s "${LOG_FILE}" ]; then
        grep -vxF -f "${LOG_FILE}" "${CURRENT_LOG_DIR}/train_stdout.txt" >> "${LOG_FILE}" || true
    elif [ -s "${CURRENT_LOG_DIR}/train_stdout.txt" ]; then
        cat "${CURRENT_LOG_DIR}/train_stdout.txt" >> "${LOG_FILE}"
    fi
    echo "train exit=${train_exit}" | tee -a "${LOG_FILE}"
fi

sleep 5
if [ -n "${MONITOR_GPU_PID}" ] && kill -0 "${MONITOR_GPU_PID}" 2>/dev/null; then
    echo "run cmd: kill -9 ${MONITOR_GPU_PID} (rank=${RANK})"
    kill -9 "${MONITOR_GPU_PID}" 2>/dev/null || true
elif [ -n "${MONITOR_GPU_SCRIPT:-}" ]; then
    pkill -f "${MONITOR_GPU_SCRIPT}" 2>/dev/null || true
fi

if [ "${RANK}" = "0" ]; then

    sleep 3
    if ls "${CURRENT_LOG_DIR}"/memory_rank*.txt >/dev/null 2>&1; then
        cat "${CURRENT_LOG_DIR}"/memory_rank*.txt > "${CURRENT_LOG_DIR}/memory.txt" || true
        MEMORY_LOG_FILE="${CURRENT_LOG_DIR}/memory.txt"
    fi

    last_line() {
        local v="$1"
        v="${v//$'\r'/}"
        printf '%s\n' "${v}" | awk 'NF { last = $0 } END { print last }'
    }

    torchrun_str=$(grep -E '^torchrun --' "${LOG_FILE}" 2>/dev/null | head -1 || true)
    if [ -z "${torchrun_str}" ]; then
        torchrun_str=$(grep -m1 'torchrun --nproc' "${LOG_FILE}" 2>/dev/null || true)
    fi
    if [ -z "${torchrun_str}" ]; then
        torchrun_str=$(grep -m1 'torchrun' "${LOG_FILE}" 2>/dev/null || true)
    fi

    nproc_per_node=$(echo ${torchrun_str} | grep -oP '\-\-nproc_per_node\s+\K\d+' | tr -d '\n\r' | head -1)
    NUM_LAYERS=$(awk '$1 == "num_layers" {print $NF; exit}' "${LOG_FILE}")
    if [ -z "${NUM_LAYERS}" ]; then
        NUM_LAYERS=$(echo ${torchrun_str} | grep -oP '\-\-num\-layers\s+\K\d+' | head -1)
    fi
    if [ -z "${NUM_LAYERS}" ]; then
        case "${MODEL_SIZE}" in
            70B|70b) NUM_LAYERS=80 ;;
            8B|8b) NUM_LAYERS=32 ;;
            0.5B|0.5b) NUM_LAYERS=24 ;;
            A21B|236B|236b|A236B) NUM_LAYERS=60 ;;
            A2.4B|16B|16b|A16B) NUM_LAYERS=27 ;;
            *) NUM_LAYERS=27 ;;
        esac
    fi
    NNODES=$(echo ${torchrun_str} | grep -oP '\-\-nnodes\s+\K\d+' | tr -d '\n\r' | head -1)
    NNODES=${NNODES:-${WORLD_SIZE:-1}}
    nproc_per_node=${nproc_per_node:-${GPUS_PER_NODE:-8}}
    NPROC=$(( ${NNODES} * ${nproc_per_node} ))
    PR=fp32
    if echo "${torchrun_str}" | grep -q -- "--bf16"; then
        PR=bf16
    elif echo "${torchrun_str}" | grep -q -- "--fp16"; then
        PR=fp16
    fi
    GLOBAL_BATCH_SIZE=$(echo ${torchrun_str} | grep -oP '\-\-global\-batch\-size \K\d+' | head -1)
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${_LAUNCH_GBS:-1024}}
    SEQ_LENGTH=$(echo ${torchrun_str} | grep -oP '\-\-seq\-length \K\d+' | head -1)
    SEQ_LENGTH=${SEQ_LENGTH:-${SEQ_LEN:-4096}}

    MICRO_BATCH_SIZE=$(echo ${torchrun_str} | grep -oP '\-\-micro\-batch\-size \K\d+' | head -1)
    TP=$(echo ${torchrun_str} | grep -oP '\-\-tensor\-model\-parallel\-size \K\d+' | head -1)
    PP=$(echo ${torchrun_str} | grep -oP '\-\-pipeline\-model\-parallel\-size \K\d+' | head -1)
    CP=$(echo ${torchrun_str} | grep -oP '\-\-context\-parallel\-size \K\d+' | head -1)
    # If training died before torchrun cmdline was logged, keep the knobs we
    # actually launched with — do not invent empty MBS / TP=1 and poison PARA_FILE.
    MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${_LAUNCH_MBS:-1}}
    TP=${TP:-${_LAUNCH_TP:-1}}
    PP=${PP:-${_LAUNCH_PP:-1}}
    CP=${CP:-${_LAUNCH_CP:-1}}

    if echo "$torchrun_str" | grep -q -- "--sequence-parallel"; then
        SP="true"
    else
        SP="${SP:-false}"
    fi
    if echo "$torchrun_str" | grep -q -- "--use-distributed-optimizer"; then
        DO="true"
    else
        DO="${DO:-false}"
    fi

    OVERLAP_GRAD_REDUCE=$(echo "$torchrun_str" | grep -q -- "--overlap-grad-reduce" && echo "true" || echo "false")
    OVERLAP_PARAM_GATHER=$(echo "$torchrun_str" | grep -q -- "--overlap-param-gather" && echo "true" || echo "false")
    TP_COMM_OVERLAP=$(echo "$torchrun_str" | grep -q -- "--tp-comm-overlap" && echo "true" || echo "false")
    AC=$(grep -E "^AC is " "$LOG_FILE" | tail -1 | awk '{print $NF}')
    AC=${AC:-$(grep -o "AC is [a-z]*" "$LOG_FILE" | tail -1 | cut -d' ' -f3)}
    AC=${AC,,}
    case "${AC}" in
        selective) AC=sel ;;
        "") AC=none ;;
    esac
    AC_LAYERS=$(grep -o "AC_LAYERS is [0-9]*" "$LOG_FILE" | tail -1 | cut -d' ' -f3)
    if [ -z "${AC_LAYERS}" ]; then
        AC_LAYERS=$(grep -E "^MP_AC_LAYERS is " "$LOG_FILE" | tail -1 | awk '{print $NF}')
    fi
    if [ -z "${AC_LAYERS}" ]; then
        AC_LAYERS=$(echo "$torchrun_str" | grep -oP -- '--recompute-num-layers\s+\K\d+' | head -1)
    fi
    AC_LAYERS=${AC_LAYERS:-1}
    AC_METHOD=$(echo "$torchrun_str" | grep -oP -- '--recompute-method\s+\K\S+' | head -1)
    AC_METHOD=${AC_METHOD,,}
    if [ -z "${AC_METHOD}" ] || [ "${AC_METHOD}" = "none" ]; then
        if [ "${AC}" = "full" ]; then
            AC_METHOD=uniform
        else
            AC_METHOD=none
        fi
    fi
    # Megatron rejects --recompute-method none; never persist that with AC=full.
    if [ "${AC}" = "full" ]; then
        case "${AC_METHOD}" in
            uniform|block) ;;
            *) AC_METHOD=uniform ;;
        esac
    fi
    AC_MODULES=$(echo "$torchrun_str" | grep -oP -- '-{1,2}recompute-modules\s+\K.*?(?=\s+--|\s*$)' | head -1 | tr ',' ' ' | xargs)
    if [ -z "${AC_MODULES}" ] || [ "${AC_MODULES}" = "none" ]; then
        AC_MODULES=$(grep -E "^AC_MODULES is " "$LOG_FILE" | tail -1 | awk '{print $NF}')
    fi
    AC_MODULES=${AC_MODULES:-none}
    # CSV-safe: spaces ok; strip commas
    AC_MODULES=$(echo "${AC_MODULES}" | tr ',' ' ' | xargs)

    FUSED_CROSS_ENTROPY=$(echo "$torchrun_str" | grep -q -- "--cross-entropy-loss-fusion" && echo "true" || echo "false")

    FIRST_PP_NUM_LAYERS=$(echo "$torchrun_str" | grep -oP -- "--decoder-first-pipeline-num-layers\s+\K\d+" || echo "None")
    LAST_PP_NUM_LAYERS=$(echo "$torchrun_str" | grep -oP -- "--decoder-last-pipeline-num-layers\s+\K\d+" || echo "None")
    FIRST_PP_NUM_LAYERS=$(last_line "${FIRST_PP_NUM_LAYERS}")
    LAST_PP_NUM_LAYERS=$(last_line "${LAST_PP_NUM_LAYERS}")
    FIRST_PP_NUM_LAYERS=${FIRST_PP_NUM_LAYERS:-None}
    LAST_PP_NUM_LAYERS=${LAST_PP_NUM_LAYERS:-None}
    if [ "${PP}" = "1" ]; then
        FIRST_PP_NUM_LAYERS=None
        LAST_PP_NUM_LAYERS=None
    fi

    if is_moe_model; then
        NUM_EXPERTS=$(awk '$1 == "num_experts" {print $NF; exit}' "${LOG_FILE}")
        if [ -z "${NUM_EXPERTS}" ] || [ "${NUM_EXPERTS}" = "None" ]; then
            NUM_EXPERTS=$(echo ${torchrun_str} | grep -oP '\-\-num\-experts\s+\K\d+' | head -1)
        fi
        if [ -z "${NUM_EXPERTS}" ]; then
            if [ "${MODEL_SIZE}" = "A21B" ]; then NUM_EXPERTS=160; else NUM_EXPERTS=64; fi
        fi
        EP=$(echo ${torchrun_str} | grep -oP '\-\-expert\-model\-parallel\-size \K\d+' | head -1)
        EP=${EP:-${_LAUNCH_EP:-1}}
    fi

    if [ -z "${torchrun_str}" ]; then
        SP=${_LAUNCH_SP:-${SP:-false}}
        DO=${_LAUNCH_DO:-${DO:-false}}
        AC=${_LAUNCH_AC:-${AC:-none}}
        AC_METHOD=${_LAUNCH_AC_METHOD:-${AC_METHOD:-none}}
        AC_LAYERS=${_LAUNCH_AC_LAYERS:-${AC_LAYERS:-1}}
        AC_MODULES=${_LAUNCH_AC_MODULES:-${AC_MODULES:-none}}
        OVERLAP_GRAD_REDUCE=${_LAUNCH_OGR:-${OVERLAP_GRAD_REDUCE:-false}}
        OVERLAP_PARAM_GATHER=${_LAUNCH_OPG:-${OVERLAP_PARAM_GATHER:-false}}
        TP_COMM_OVERLAP=${_LAUNCH_TPCO:-${TP_COMM_OVERLAP:-false}}
        FUSED_CROSS_ENTROPY=${_LAUNCH_FCE:-${FUSED_CROSS_ENTROPY:-false}}
        FIRST_PP_NUM_LAYERS=${_LAUNCH_PP0:-${FIRST_PP_NUM_LAYERS:-None}}
        LAST_PP_NUM_LAYERS=${_LAUNCH_PPN:-${LAST_PP_NUM_LAYERS:-None}}
    fi

    last_rank=$(( ${NNODES:-${WORLD_SIZE:-1}} - 1 ))
    if [ "${last_rank}" -lt 0 ] 2>/dev/null; then
        last_rank=0
    fi
    logfile_lastnode="${MEGATRON_LOG_DIR}/${last_rank}.txt"
    if [ ! -f "${logfile_lastnode}" ]; then
        logfile_lastnode="${LOG_FILE}"
    fi

    # Last logged iteration only. Train scripts run 3 iters; iter 3 is the steady step.
    tflops=$(grep -oP 'throughput per GPU \(TFLOP/s/GPU\): \K\d+\.\d+' "${logfile_lastnode}" 2>/dev/null | tail -n 1)
    tflops=${tflops:-0}
    if ! awk -v t="${tflops:-0}" 'BEGIN {exit !(t > 0)}'; then
        tflops=$(grep -oP 'throughput \(tflops/sec/GPU\) : \K\d+\.\d+' "${logfile_lastnode}" 2>/dev/null | tail -n 1)
        tflops=${tflops:-0}
    fi
    peak_flops=${GPU_PEAK_TFLOPS:-312}
    mfu=$(awk -v t="${tflops:-0}" -v p="${peak_flops}" 'BEGIN {if (p>0) print t/p; else print 0}')
    iter_time=$(grep -oP 'elapsed time per iteration \(ms\): \K\d+\.\d+' "${logfile_lastnode}" 2>/dev/null | tail -n 1)
    iter_time=$(awk -v ms="${iter_time:-0}" 'BEGIN {if (ms>0) printf "%.2f", ms/1000; else print 0}')
    tgs=$(grep -oP 'throughput \(token/sec/GPU\) : \K\d+\.\d+' "${logfile_lastnode}" 2>/dev/null | tail -n 1)
    tgs=${tgs:-0}
    if ! awk -v t="${tgs:-0}" 'BEGIN {exit !(t > 0)}'; then
        _seq_for_tgs=${SEQ_LENGTH:-${SEQ_LEN:-0}}
        if awk -v it="${iter_time:-0}" -v n="${NPROC:-0}" -v g="${GLOBAL_BATCH_SIZE:-0}" -v s="${_seq_for_tgs}" \
            'BEGIN {exit !(it > 0 && n > 0 && g > 0 && s > 0)}'; then
            tgs=$(awk -v g="${GLOBAL_BATCH_SIZE}" -v s="${_seq_for_tgs}" -v it="${iter_time}" -v n="${NPROC}" \
                'BEGIN {printf "%.1f", g * s / (it * n)}')
        else
            tgs=0
        fi
    fi

    if awk -v tflops="${tflops}" 'BEGIN {exit !(tflops > 0)}'; then
        result=success
        mem_ave=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $3}' | grep -oP '\d+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        mem_max=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $3}' | grep -oP '\d+' | awk 'BEGIN{ max = 0} {if ($1 > max) max = $1} END{print max}')
        mem_limit=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $4}' | grep -oP '\d+')
        IFS=' '
        mem_limit=$(echo ${mem_limit} | sort -n | head -n1)
        gpu_util_ave=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $5}' | grep -oP '\d+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        gpu_util_max=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $5}' | grep -oP '\d+' | awk 'BEGIN{ max = 0} {if ($1 > max) max = $1} END{print max}')
    else
        echo "ERROR!!! tflops is ${tflops}"
        result=fail
        iter_time=NULL
        tflops=NULL
        tgs=NULL
        mfu=NULL
        mem_ave=NULL
        mem_max=NULL
        mem_limit=NULL
        gpu_util_ave=NULL
        gpu_util_max=NULL
        echo "save error info"
        CURRENT_ERROR_LOG_FILE=${CURRENT_LOG_DIR}/monitor_error.txt
        rank_tmp=0
        _err_ranks=${NNODES:-${WORLD_SIZE:-1}}
        while [ ${rank_tmp} -lt ${_err_ranks} ]; do
            log_file_tmp="${MEGATRON_LOG_DIR}/${rank_tmp}.txt"
            bash ${SCRIPT_PATH}/monitor/extract_error.sh ${log_file_tmp} ${CURRENT_ERROR_LOG_FILE}
            rank_tmp=$[ ${rank_tmp} + 1 ]
        done
        cp ${CURRENT_ERROR_LOG_FILE} ${ERROR_LOG_FILE}
    fi

    echo "save data to csv"
    title_list="MICRO_BATCH_SIZE,TP,PP,CP,SP"
    data_list="${MICRO_BATCH_SIZE},${TP},${PP},${CP},${SP}"
    if is_moe_model; then
        title_list="${title_list},EP"
        data_list="${data_list},${EP}"
    fi
    title_list="${title_list},DO,AC,AC_METHOD,AC_LAYERS,AC_MODULES,OVERLAP_GRAD_REDUCE,OVERLAP_PARAM_GATHER,TP_COMM_OVERLAP,FUSED_CROSS_ENTROPY,FIRST_PP_NUM_LAYERS,LAST_PP_NUM_LAYERS,Result,TFLOPS,MFU,MemAve,MemMax,GpuUtilAve,GpuUtilMax,Log Director,Reasons for failure"
    data_list="${data_list},${DO},${AC},${AC_METHOD},${AC_LAYERS},${AC_MODULES},${OVERLAP_GRAD_REDUCE},${OVERLAP_PARAM_GATHER},${TP_COMM_OVERLAP},${FUSED_CROSS_ENTROPY},${FIRST_PP_NUM_LAYERS},${LAST_PP_NUM_LAYERS},${result},${tflops},${mfu},${mem_ave},${mem_max},${gpu_util_ave},${gpu_util_max},${ITER_CUR},NULL"
    if [ ! -f ${REPORT_FILE} ]; then
        echo "${title_list}" > ${REPORT_FILE}
    fi
    echo "${data_list}" >> ${REPORT_FILE}

    echo "save fix parameters"
    if [ ! -f "${FIXED_PARA_FILE}" ]; then
        if [[ "$mem_limit" == "NULL" ]] || [ -z "${mem_limit}" ]; then
            mem_limit=${GPU_MEMORY_LIMIT_MB:-81920}
        fi
        echo "Model Type: ${MODEL_TYPE}" >> ${FIXED_PARA_FILE}
        echo "Model Size: ${MODEL_SIZE}" >> ${FIXED_PARA_FILE}
        echo "NUM_LAYERS: ${NUM_LAYERS}" >> ${FIXED_PARA_FILE}
        if is_moe_model; then
            echo "NUM_EXPERTS: ${NUM_EXPERTS}" >> ${FIXED_PARA_FILE}
        fi
        echo "Precison: ${PR}" >> ${FIXED_PARA_FILE}
        echo "Global Batch Size: ${GLOBAL_BATCH_SIZE}" >> ${FIXED_PARA_FILE}
        echo "Sequence Length: ${SEQ_LENGTH}" >> ${FIXED_PARA_FILE}
        echo "Number of Nodes: ${NNODES}" >> ${FIXED_PARA_FILE}
        echo "Number of GPUs: ${NPROC}" >> ${FIXED_PARA_FILE}
        echo "GPU memory limit per card: ${mem_limit}" >> ${FIXED_PARA_FILE}
    fi

    echo "save parameters"
    CURRENT_PARA_FILE=${CURRENT_LOG_DIR}/parameters.txt
    {
        echo "MICRO_BATCH_SIZE: ${MICRO_BATCH_SIZE}"
        echo "TP: ${TP}"
        echo "PP: ${PP}"
        echo "CP: ${CP}"
        echo "SP: ${SP,,}"
        if is_moe_model; then
            echo "EP: ${EP}"
        fi
        echo "DO: ${DO,,}"
        echo "AC: ${AC,,}"
        echo "AC_METHOD: ${AC_METHOD,,}"
        echo "AC_LAYERS: ${AC_LAYERS}"
        echo "AC_MODULES: ${AC_MODULES}"
        echo "OVERLAP_GRAD_REDUCE: ${OVERLAP_GRAD_REDUCE,,}"
        echo "OVERLAP_PARAM_GATHER: ${OVERLAP_PARAM_GATHER,,}"
        echo "TP_COMM_OVERLAP: ${TP_COMM_OVERLAP,,}"
        echo "FUSED_CROSS_ENTROPY: ${FUSED_CROSS_ENTROPY,,}"
        echo "FIRST_PP_NUM_LAYERS: ${FIRST_PP_NUM_LAYERS}"
        echo "LAST_PP_NUM_LAYERS: ${LAST_PP_NUM_LAYERS}"
    } > ${CURRENT_PARA_FILE}

    if [ ! -f "${PARA_FILE}" ]; then
        cp ${CURRENT_PARA_FILE} ${PARA_FILE}
    fi

    # record best result (MFU as percent in best_parameters.json, matching Meta schema)
    BEST_PARAMETERS_FILE=${BEST_PARAMETERS_FILE:-"${OUTPUT_PATH}/best_parameters.json"}
    if [ "${result}" = "success" ]; then
        mfu_pct=$(awk -v m="${mfu:-0}" 'BEGIN {print (m > 1) ? m : m * 100}')
        update_best="false"
        if [ ! -f "${BEST_PARAMETERS_FILE}" ]; then
            update_best="true"
        else
            best_mfu=$(python3 - "${BEST_PARAMETERS_FILE}" << 'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("MFU", 0))
PY
)
            if awk "BEGIN {exit !(${mfu_pct} > ${best_mfu:-0})}"; then
                update_best="true"
            fi
        fi
        if [ "${update_best}" = "true" ]; then
            SEARCH_START_FILE=${SEARCH_START_FILE:-"${OUTPUT_PATH}/search_start.epoch"}
            if [ -z "${SEARCH_START_EPOCH:-}" ] && [ -f "${SEARCH_START_FILE}" ]; then
                SEARCH_START_EPOCH=$(tr -d '[:space:]' < "${SEARCH_START_FILE}")
            fi
            if [ -z "${SEARCH_START_EPOCH:-}" ]; then
                SEARCH_START_EPOCH=$(date +%s)
                echo "${SEARCH_START_EPOCH}" > "${SEARCH_START_FILE}"
            fi
            elapsed_time=$(( $(date +%s) - SEARCH_START_EPOCH ))
            _seq_for_best=${SEQ_LENGTH:-${SEQ_LEN:-0}}
            python3 "${SCRIPT_PATH}/monitor/update_best_parameters.py" \
                --para-file "${CURRENT_PARA_FILE}" \
                --output-file "${BEST_PARAMETERS_FILE}" \
                --output-path "${OUTPUT_PATH}" \
                --mfu "${mfu_pct}" \
                --tflops "${tflops}" \
                --tgs "${tgs:-0}" \
                --iter-time "${iter_time}" \
                --iter "${ITER_CUR}" \
                --elapsed "${elapsed_time}" \
                --gbs "${GLOBAL_BATCH_SIZE:-0}" \
                --seq "${_seq_for_best}" \
                --nproc "${NPROC:-0}"
        fi
    fi

    echo ${result} > ${RESULT_FILE}
    echo ${result} > ${CURRENT_LOG_DIR}/monitor_result.txt

    echo "run ${train_script} ${result}"

fi
