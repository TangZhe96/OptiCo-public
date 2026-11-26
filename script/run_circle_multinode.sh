#!/bin/bash
set +e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
export OPTICO_ROOT_PATH=${OPTICO_ROOT_PATH:-$(cd "${SCRIPT_DIR}/.." && pwd)}

export NCCL_DEBUG=ERROR

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
sysctl -w net.ipv4.tcp_keepalive_time=30
sysctl -w net.ipv4.tcp_keepalive_intvl=10
sysctl -w net.ipv4.tcp_keepalive_probes=3

MODEL_TYPE=${1:-${MODEL_TYPE:-${MODEL:-"qwen2.5"}}}
MODEL_SIZE=${2:-${MODEL_SIZE:-"0.5B"}}
NUMBER_OF_CYCLES=${3:-${NUMBER_OF_CYCLES:-20}}

_mt=$(printf '%s' "${MODEL_TYPE:-${MODEL:-}}" | tr '[:upper:]' '[:lower:]')
case "${_mt}" in
    deepseek_v2|deepseek*)
        case "${MODEL_SIZE}" in
            16B|16b|A16B) MODEL_SIZE=A2.4B ;;
            236B|236b|A236B) MODEL_SIZE=A21B ;;
        esac
        ;;
    021)
        case "${MODEL_SIZE}" in
            A21B|A21b|a21b|236b|A236B) MODEL_SIZE=236B ;;
            32b) MODEL_SIZE=32B ;;
            mini) MODEL_SIZE=MINI ;;
        esac
        ;;
esac
unset _mt

export MODEL_TYPE=${MODEL_TYPE}
export MODEL=${MODEL:-${MODEL_TYPE}}
export MODEL_SIZE=${MODEL_SIZE}
export NUMBER_OF_CYCLES=${NUMBER_OF_CYCLES}
export PLATFORM=${PLATFORM:-A100}

if [ -z "${NUM_NODES:-}" ] && [ -n "${WORLD_SIZE:-}" ]; then
    export NUM_NODES="${WORLD_SIZE}"
fi
if [ -n "${NUM_NODES:-}" ]; then
    export WORLD_SIZE="${NUM_NODES}"
fi
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

export KEY=${KEY:-${API_KEY:-}}
export API_KEY=${API_KEY:-${KEY:-}}

export PROMPT_HOME_PATH=${PROMPT_HOME_PATH:-"${OPTICO_ROOT_PATH}/prompt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${OPTICO_ROOT_PATH}/output_optico"}
export OUTPUT_PATH=${OUTPUT_PATH:-"${OUTPUT_DIR}/output_${MODEL_TYPE}_${MODEL_SIZE}_n${WORLD_SIZE}"}
export REPORT_FILE="${OUTPUT_PATH}/report.csv"
export PARA_FILE="${OUTPUT_PATH}/parameters.txt"
export FIXED_PARA_FILE="${OUTPUT_PATH}/fixed_parameters.txt"
export MONITOR_RESULT_FILE="${OUTPUT_PATH}/monitor_result.txt"
export ERROR_LOG_FILE="${OUTPUT_PATH}/monitor_error.txt"
export MODEL_PATH=${MODEL_PATH:-"/workspace/model/DeepSeek-R1-0528-Qwen3-8B"}

mkdir -p "${OUTPUT_PATH}"

monitor_cmd="bash ${SCRIPT_DIR}/monitor/run_monitor_multinode.sh"
loc_error_cmd="bash ${SCRIPT_DIR}/analyzer/run_analyze.sh loc_error"
auto_analyze_cmd="bash ${SCRIPT_DIR}/analyzer/run_analyze.sh auto"

iter_cur=1
if [ -e ${REPORT_FILE} ]; then
    iter_cur=$(grep -v "^#" ${REPORT_FILE} | tail -n +2 | grep -v "^$" | wc -l)
    iter_cur=$[ ${iter_cur} + 1 ]
fi

iter_max=$[ ${iter_cur} + ${NUMBER_OF_CYCLES} ]

retry_num=0
retry_max=2

while [ ${iter_cur} -lt ${iter_max} ]; do
    echo "[${iter_cur}] ---------------------------- run_circle start ----------------------------"
    export ITER_CUR=${iter_cur}
    echo "[${iter_cur}] run cmd: ${monitor_cmd}"
    eval ${monitor_cmd}

    CURRENT_RESULT_FILE="${OUTPUT_PATH}/${iter_cur}/monitor_result.txt"

    if [ "${RANK}" -gt 0 ] 2>/dev/null; then
        while true; do
            if [ -f "$CURRENT_RESULT_FILE" ]; then
                break
            fi
            sleep 1
        done
    fi

    monitor_result=$(cat "${CURRENT_RESULT_FILE}")
    if [[ "${monitor_result}" == "success" ]]; then
        echo "[${iter_cur}] monitor success"
    elif [[ "${monitor_result}" == "fail" ]]; then
        echo "[${iter_cur}] monitor fail"

        echo "[${iter_cur}] loc_error start"

        if [ "${RANK}" -eq 0 ]; then
            eval ${loc_error_cmd}
        else
            while true; do
                if [ -f "${OUTPUT_PATH}/${iter_cur}/loc_error.txt" ]; then
                    break
                fi
                sleep 1
            done
        fi

        echo "[${iter_cur}] loc_error finish"
        loc_error=$(cat ${OUTPUT_PATH}/${iter_cur}/loc_error.txt)
        echo "cause of failure: ${loc_error}"

        if [[ "${loc_error}" == *"RETRY"* ]] && [[ ${retry_num} -lt ${retry_max} ]]; then
            echo "[${iter_cur}] KEEP the parameters and restart the training"
            retry_num=$[ ${retry_num} + 1 ]
            echo "[${iter_cur}] Number of retries: ${retry_num} ..."
            sleep 5
            iter_cur=$[ ${iter_cur} + 1 ]
            iter_max=$[ ${iter_max} + 1 ]
            continue
        else
            if [[ "${loc_error}" == *"RETRY"* ]]; then
                echo "[${iter_cur}] Number of retries ${retry_num} exceed the limit ${retry_max}. Stop retry."
            else
                echo "[${iter_cur}] modify the parameters and restart the training"
            fi
        fi

    else
        echo "[${iter_cur}] ERROR!!!! unknown monitor result: ${monitor_result}. Stop circle."
        exit
    fi

    retry_num=0

    analyze_result_file="${OUTPUT_PATH}/${iter_cur}/analyze_result.txt"
    if [ "${RANK}" -eq 0 ]; then
        echo "[${iter_cur}] give_sugg and change_para start"
        if ! eval ${auto_analyze_cmd}; then
            echo "fail" >> ${analyze_result_file}
            echo "[${iter_cur}] parameter validation failed. Stop circle."
            exit 1
        fi
        echo "[${iter_cur}] give_sugg and change_para finish"
        echo "success" >> ${analyze_result_file}
    else
        while true; do
            if [ -f "$analyze_result_file" ]; then
                analyze_result=$(cat ${analyze_result_file})
                if [ "$analyze_result" = "success" ]; then
                    break
                fi
                if [ "$analyze_result" = "fail" ]; then
                    echo "[${iter_cur}] parameter validation failed. Stop circle."
                    exit 1
                fi
            fi
            sleep 5
        done
    fi

    echo "[${iter_cur}] ============================ run_circle finish ============================"

    iter_cur=$[ ${iter_cur} + 1 ]

done
