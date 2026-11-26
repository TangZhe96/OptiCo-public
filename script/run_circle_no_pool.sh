#!/bin/bash
set +e

# ROOT_DIR=$(cd "$(dirname "$0")"; pwd)
export NCCL_DEBUG=ERROR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
sysctl -w net.ipv4.tcp_keepalive_time=30
sysctl -w net.ipv4.tcp_keepalive_intvl=10
sysctl -w net.ipv4.tcp_keepalive_probes=3


MODEL_TYPE=${1:-"qwen2.5"} # qwen2.5  llama3 llama2 qwen3-moe gpt3
MODEL_SIZE=${2:-"0.5B"} # 0.5B  7B  8B A0.6B
NUMBER_OF_CYCLES=${3:-20}


export MODEL_TYPE=${MODEL_TYPE}
export MODEL_SIZE=${MODEL_SIZE}

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}


PROMPT_HOME_PATH=/workspace/code/OptiCo/prompt
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/code/OptiCo/output_ablation/no_pool"}
export OUTPUT_PATH=${OUTPUT_PATH:-"${OUTPUT_DIR}/output_${MODEL_TYPE}_${MODEL_SIZE}_n${WORLD_SIZE}"}
export REPORT_FILE="${OUTPUT_PATH}/report.csv"
export REPORT_FILE_LAST="${OUTPUT_PATH}/report_last.csv"
export PARA_FILE="${OUTPUT_PATH}/parameters.txt"
export FIXED_PARA_FILE="${OUTPUT_PATH}/fixed_parameters.txt"
export MONITOR_RESULT_FILE="${OUTPUT_PATH}/monitor_result.txt"
export ERROR_LOG_FILE="${OUTPUT_PATH}/monitor_error.txt"
export MODEL_PATH="/workspace/model/DeepSeek-R1-0528-Qwen3-8B"


monitor_cmd="bash monitor/run_monitor_no_pool.sh"
loc_error_cmd="bash analyzer/run_analyze_no_pool.sh loc_error"
auto_analyze_cmd="bash analyzer/run_analyze_no_pool.sh auto"

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

    if [ $RANK > 0 ]; then
        while true; do
            if [ -f "$MONITOR_RESULT_FILE" ]; then
                break
            fi
            sleep 1
        done
    fi
    
    monitor_result=$(cat "${MONITOR_RESULT_FILE}")
    if [[ "${monitor_result}" == "success" ]]; then
        echo "[${iter_cur}] monitor success"
    elif [[ "${monitor_result}" == "fail" ]]; then
        echo "[${iter_cur}] monitor fail"
        
        echo "[${iter_cur}] loc_error start"
        
        if [ $RANK == 0 ]; then
            eval ${loc_error_cmd}
        else
            while true; do
                if [ -f "${OUTPUT_PATH}/loc_error.txt" ]; then
                    break
                fi
                sleep 1
            done
        fi

        echo "[${iter_cur}] loc_error finish"
        loc_error=$(cat ${OUTPUT_PATH}/loc_error.txt)
        echo "cause of failure: ${loc_error}"

        # retry
        if [[ "${loc_error}" == *"RETRY"* ]] && [[ ${retry_num} -lt ${retry_max} ]]; then
            if [ ${retry_num} -lt ${retry_max} ]; then
                echo "[${iter_cur}] KEEP the parameters and restart the training"
                retry_num=$[ ${retry_num} + 1 ]
                echo "[${iter_cur}] Number of retries: ${retry_num} ..."
                sleep 5
                iter_cur=$[ ${iter_cur} + 1 ]
                iter_max=$[ ${iter_max} + 1 ]
                continue
            else
                echo "[${iter_cur}] Number of retries ${retry_num} exceed the limit ${retry_max}. Stop retry."
            fi
        else
            echo "[${iter_cur}] modify the parameters and restart the training"
        fi

    else
        echo "[${iter_cur}] ERROR!!!! unknown monitor result: ${monitor_result}. Stop circle."
        exit
    fi
    
    retry_num=0

    analyze_result_file="${OUTPUT_PATH}/${iter_cur}/analyze_result.txt"
    if [ $RANK = 0 ]; then
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

