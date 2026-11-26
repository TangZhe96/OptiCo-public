#!/bin/bash

MODEL_TYPE=${MODEL_TYPE:-"qwen2.5"}
MODEL_SIZE=${MODEL_SIZE:-"0.5B"}

PROMPT_HOME_PATH=/workspace/dataset/OptiCo
export OUTPUT_PATH="/workspace/OptiCo/output_answer_quest"
export REPORT_FILE="${OUTPUT_PATH}/report.csv"
export PARA_FILE="${OUTPUT_PATH}/parameters.txt"
export MONITOR_RESULT_FILE="${OUTPUT_PATH}/monitor_result.txt"
export ERROR_LOG_FILE="${OUTPUT_PATH}/monitor_error.txt"

MODEL_PATH=/workspace/model/DeepSeek-R1-Distill-Qwen-14B


source /workspace/venv_vllm/bin/activate 

export CUDA_VISIBLE_DEVICES=0,1,2,3

iter_cur=1
if [ -e ${REPORT_FILE} ]; then
    iter_cur=$(grep -v "^#" ${REPORT_FILE} | tail -n +2 | grep -v "^$" | wc -l)
    export ITER_CUR=${iter_cur}
fi

echo "[${iter_cur}] ---------------------------- answer_quest start ----------------------------"

python /workspace/OptiCo/script/analyzer/analyzer.py \
    --prompt-answer-quest /workspace/dataset//time_consuming.txt \
    --time-consuming-table /workspace/dataset//time_consuming.csv \
    --model-path ${MODEL_PATH} \
    --output-path ${OUTPUT_PATH} \
    --max-length 8192 --tp 4 --run-type answer_quest

echo "[${iter_cur}] ============================ answer_quest finish ============================"

deactivate