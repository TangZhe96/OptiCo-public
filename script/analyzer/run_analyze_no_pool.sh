#!/bin/bash
set +e

RUN_TYPE=${1:-auto} # auto / give_sugg / change_para / loc_error / answer_quest

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


MODEL_TYPE=${MODEL_TYPE:-"qwen2.5"}
MODEL_SIZE=${MODEL_SIZE:-"0.5B"}

PROMPT_HOME_PATH=/workspace/code/OptiCo/prompt
OUTPUT_PATH=${OUTPUT_PATH:-"/workspace/code/OptiCo/output_${MODEL_TYPE}_${MODEL_SIZE}"}
REPORT_FILE=${REPORT_FILE:-"${OUTPUT_PATH}/report.csv"}
REPORT_FILE_LAST=${REPORT_FILE_LAST:-"${OUTPUT_PATH}/report_last.csv"}
PARA_FILE=${PARA_FILE:-"${OUTPUT_PATH}/parameters.txt"}
FIXED_PARA_FILE=${FIXED_PARA_FILE:-"${OUTPUT_PATH}/fixed_parameters.txt"}
MONITOR_RESULT_FILE=${MONITOR_RESULT_FILE:-"${OUTPUT_PATH}/monitor_result.txt"}
ERROR_LOG_FILE=${ERROR_LOG_FILE:-"${OUTPUT_PATH}/monitor_error.txt"}
MODEL_PATH=${MODEL_PATH:-"/workspace/model/DeepSeek-R1-Distill-Qwen-14B"}

iter_cur=1
if [ -e ${REPORT_FILE} ]; then
    iter_cur=$(grep -v "^#" ${REPORT_FILE} | tail -n +2 | grep -v "^$" | wc -l)
    export ITER_CUR=${iter_cur}
fi

TP=8

add_args=""

give_sugg_cot_filepath=${PROMPT_HOME_PATH}/give_sugg_cot.txt
change_para_filepath=${PROMPT_HOME_PATH}/change_para.txt
give_sugg_filepath=${PROMPT_HOME_PATH}/give_sugg.txt

if [ ${RUN_TYPE} = auto ]; then
    add_args="
        --prompt-give-sugg ${give_sugg_filepath} \
        --cot-give-sugg ${give_sugg_cot_filepath} \
        --monitor-result-file ${MONITOR_RESULT_FILE} \
        --error-file ${ERROR_LOG_FILE} \
        --prompt-change-para ${change_para_filepath} \
    "
elif [ ${RUN_TYPE} = give_sugg ]; then
    add_args="
        --prompt-give-sugg ${give_sugg_filepath} \
        --cot-give-sugg ${give_sugg_cot_filepath} \
        --monitor-result-file ${MONITOR_RESULT_FILE} \
        --error-file ${ERROR_LOG_FILE} \
    "
elif [ ${RUN_TYPE} = change_para ]; then
    add_args="
        --prompt-change-para ${change_para_filepath} \
    "
elif [ ${RUN_TYPE} = loc_error ]; then
    add_args="
        --prompt-loc-error ${PROMPT_HOME_PATH}/loc_error.txt \
        --error-file ${ERROR_LOG_FILE} \
    "

elif [ ${RUN_TYPE} = answer_quest ]; then
    add_args="
        --prompt-answer-quest /workspace/dataset//time_consuming.txt \
        --time-consuming-table /workspace/dataset//time_consuming.csv \
    "
fi

common_args="
    --report-table ${REPORT_FILE_LAST} \
    --para-file ${PARA_FILE} \
    --fixed-para-file ${FIXED_PARA_FILE} \
    --model-path ${MODEL_PATH} \
    --output-path ${OUTPUT_PATH} \
    --max-length 65536 \
    --tp ${TP} \
    --run-type ${RUN_TYPE}
"

echo "[${iter_cur}] ---------------------------- run ${RUN_TYPE} start ----------------------------"

python /workspace/code/OptiCo/script/analyzer/analyzer.py ${add_args} ${common_args}

_analyze_rc=$?
echo "[${iter_cur}] ============================ run ${RUN_TYPE} finish ============================"
exit ${_analyze_rc}
