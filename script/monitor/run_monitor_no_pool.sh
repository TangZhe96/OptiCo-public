#!/bin/bash
set +e

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning,ignore::ResourceWarning,ignore::DeprecationWarning"
export NCCL_WARN=ERROR
export NCCL_TIMEOUT=300

MODEL_TYPE=${MODEL_TYPE:-"qwen2.5"} # qwen2.5 llama3 llama2 qwen3-moe
MODEL_SIZE=${MODEL_SIZE:-"0.5B"} # 0.5B 7B 8B A0.6B

# log_file
OUTPUT_PATH=${OUTPUT_PATH:-"/workspace/code/OptiCo/output_${MODEL_TYPE}_${MODEL_SIZE}"}
REPORT_FILE=${REPORT_FILE:-"${OUTPUT_PATH}/report.csv"}
REPORT_FILE_LAST=${REPORT_FILE_LAST:-"${OUTPUT_PATH}/report_last.csv"}
PARA_FILE=${PARA_FILE:-"${OUTPUT_PATH}/parameters.txt"}
FIXED_PARA_FILE=${FIXED_PARA_FILE:-"${OUTPUT_PATH}/fixed_parameters.txt"}
RESULT_FILE=${RESULT_FILE:-"${OUTPUT_PATH}/monitor_result.txt"}
ERROR_LOG_FILE=${ERROR_LOG_FILE:-"${OUTPUT_PATH}/monitor_error.txt"}

ITER_CUR=${ITER_CUR:-"MONITOR_TEST"}
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
# LOG_DIR_NAME=${ITER_CUR}/monitor_log_${current_time}
CURRENT_LOG_DIR=${OUTPUT_PATH}/${ITER_CUR}
mkdir -p ${CURRENT_LOG_DIR}
MEGATRON_LOG_DIR="${CURRENT_LOG_DIR}/megatron_log"
mkdir -p ${MEGATRON_LOG_DIR}
export LOG_FILE="${MEGATRON_LOG_DIR}/${RANK}.txt"
export MEMORY_LOG_FILE="${CURRENT_LOG_DIR}/memory.txt"

SCRIPT_PATH="/workspace/code/OptiCo/script"

if [ -f "${PARA_FILE}" ]; then
    echo "read parameters"
    while IFS=':' read -r key value || [ -n "$key" ]; do
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        if [ -n "$key" ]; then
            export "$key=$value"
            echo "export $key=$value"
        fi
    done < "${PARA_FILE}"
fi


# run memmory monitor
if [ "$RANK" = 0 ]; then
    echo "monitor memmory"
    MONITOR_GPU_SCRIPT="${SCRIPT_PATH}/monitor/monitor_gpu.sh"
cat > ${MONITOR_GPU_SCRIPT} <<'EOF'
#!/bin/bash
sleep 10
echo "Temp Power Memory-Usage Memory-ALL Util% " > ${MEMORY_LOG_FILE}
while true; do
    echo "$(date -d '+%Y.%m.%d-%H.%M.%S')" >> ${MEMORY_LOG_FILE}
    echo "$(nvidia-smi | grep 'Default'| awk '{print $3,$5,$9,$11,$13}')" >> ${MEMORY_LOG_FILE}
    echo -e >> ${MEMORY_LOG_FILE}
    sleep 5
done
EOF
    # echo "$(date -d '+8 hours' '+%Y.%m.%d-%H.%M.%S')" >> ${MEMORY_LOG_FILE}
    chmod +x ${MONITOR_GPU_SCRIPT}
    nohup ${MONITOR_GPU_SCRIPT} >/dev/null 2>&1 &
fi

# run training
if [ $MODEL_TYPE = qwen2.5 ]; then
    train_script=${SCRIPT_PATH}/train/run_qwen.sh
elif [ $MODEL_TYPE = llama3 ]; then
    train_script=${SCRIPT_PATH}/train/run_llama3.sh
elif [ $MODEL_TYPE = llama2 ]; then
    train_script=${SCRIPT_PATH}/train/run_llama2.sh
elif [ $MODEL_TYPE = qwen3-moe ]; then
    train_script=${SCRIPT_PATH}/train/run_qwen3_moe.sh
elif [ $MODEL_TYPE = gpt3 ]; then
    train_script=${SCRIPT_PATH}/train/run_gpt3.sh
fi
run_cmd="bash ${train_script}"
echo "run cmd: ${run_cmd}"
eval ${run_cmd}


# stop memory monitor
if [ $RANK = 0 ]; then

    sleep 5
    pid=$(ps -ef | grep "${MONITOR_GPU_SCRIPT}"  | head -n 1 | awk '{print $2}') 
    kill_cmd="kill -9 ${pid}"
    echo "run cmd: ${kill_cmd}"
    eval ${kill_cmd} 

    # read parameters from torchrun cmd
    torchrun_str=$(grep 'torchrun' ${LOG_FILE})
        # fixed paras
    nproc_per_node=$(echo ${torchrun_str} | grep -oP '\-\-nproc\_per\_node \K\d+\.?\d*')
    NUM_LAYERS=$(awk '$1 == "num_layers" {print $NF}' "${LOG_FILE}")
    NNODES=$(echo ${torchrun_str} | grep -oP '\-\-nnodes \K\d+\.?\d*')
    NPROC=$(( ${NNODES} * ${nproc_per_node} ))
    PR=fp32
    if [[ $(echo ${torchrun_str} | grep -oP "\-\-bf16") != "" ]]; then
        PR=bf16
    elif [[ $(echo ${torchrun_str} | grep -oP "\-\-fp16") != "" ]]; then
        PR=fp16
    fi
    GLOBAL_BATCH_SIZE=$(echo ${torchrun_str} | grep -oP '\-\-global\-batch\-size \K\d+\.?\d*')
    SEQ_LENGTH=$(echo ${torchrun_str} | grep -oP '\-\-seq\-length \K\d+\.?\d*')

    # tunable paras
    MICRO_BATCH_SIZE=$(echo ${torchrun_str} | grep -oP '\-\-micro\-batch\-size \K\d+\.?\d*')
    TP=$(echo ${torchrun_str} | grep -oP '\-\-tensor\-model\-parallel\-size \K\d+\.?\d*')
    PP=$(echo ${torchrun_str} | grep -oP '\-\-pipeline\-model\-parallel\-size \K\d+\.?\d*')
    
    OVERLAP_GRAD_REDUCE=$(echo "$torchrun_str" | grep -q -- "--overlap-grad-reduce" && echo "true" || echo "false")
    OVERLAP_PARAM_GATHER=$(echo "$torchrun_str" | grep -q -- "--overlap-param-gather" && echo "true" || echo "false")
    AC=$(grep -o "AC is [a-z]*" "$LOG_FILE" | cut -d' ' -f3) # full / sel / none / offload
    AC=${AC,,}
    AC_LAYERS=$(grep -o "AC_LAYERS is [0-9]*" "$LOG_FILE" | cut -d' ' -f3)
    
    FUSED_CROSS_ENTROPY=$(echo "$torchrun_str" | grep -q -- "--cross-entropy-loss-fusion" && echo "true" || echo "false")

    FIRST_PP_NUM_LAYERS=$(echo "$torchrun_str" | grep -oP -- "--decoder-first-pipeline-num-layers\s+\K\d+" || echo "None")
    LAST_PP_NUM_LAYERS=$(echo "$torchrun_str" | grep -oP -- "--decoder-last-pipeline-num-layers\s+\K\d+" || echo "None")

    if [[ "${MODEL_TYPE,,}" == *"moe"* ]]; then
        NUM_EXPERTS=$(awk '$1 == "num_experts" {print $NF}' "${LOG_FILE}")
        EP=$(echo ${torchrun_str} | grep -oP '\-\-expert\-model\-parallel\-size \K\d+\.?\d*')
    fi

    logfile_lastnode="${MEGATRON_LOG_DIR}/$((WORLD_SIZE - 1)).txt"
    tflops=$(grep -oP 'throughput per GPU \(TFLOP/s/GPU\): \K\d+\.\d+' ${logfile_lastnode} | tail -n +3 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}') 
    mfu=$(awk "BEGIN{print ${tflops} / 312}")

    if awk -v tflops="${tflops}" 'BEGIN {exit !(tflops > 0)}'; then
        # run success
        result=success

        # count memory
        mem_ave=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $3}' | grep -oP '\d+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        mem_max=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $3}' | grep -oP '\d+' | awk 'BEGIN{ max = 0} {if ($1 > max) max = $1; fi} END{print max}')
        mem_limit=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $4}' | grep -oP '\d+')
        IFS=' '
        mem_limit=$(echo ${mem_limit} | sort -n | head -n1)
        # count gpu util
        gpu_util_ave=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $5}' | grep -oP '\d+' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
        gpu_util_max=$(grep -P 'MiB' ${MEMORY_LOG_FILE} | awk '{print $5}' | grep -oP '\d+' | awk 'BEGIN{ max = 0} {if ($1 > max) max = $1; fi} END{print max}')

    else
        # run fail
        echo "ERROR!!! tflops is ${tflops}"

        result=fail
        
        tflops=NULL
        mfu=NULL
        mem_ave=NULL
        mem_max=NULL
        mem_limit=NULL
        gpu_util_ave=NULL
        gpu_util_max=NULL
        
        # save error info
        echo "save error info"
        CURRENT_ERROR_LOG_FILE=${CURRENT_LOG_DIR}/monitor_error.txt

        rank_tmp=0
        while [ ${rank_tmp} -lt ${WORLD_SIZE} ]; do
            log_file_tmp="${MEGATRON_LOG_DIR}/${rank_tmp}.txt"
            bash ${SCRIPT_PATH}/monitor/extract_error.sh ${log_file_tmp} ${CURRENT_ERROR_LOG_FILE}
            rank_tmp=$[ ${rank_tmp} + 1 ]
        done

        cp ${CURRENT_ERROR_LOG_FILE} ${ERROR_LOG_FILE}
        # exit
    fi

    # update report
    echo "save data to csv"
    if [[ "${MODEL_TYPE,,}" == *"moe"* ]]; then
        # moe
        if [ ! -f ${REPORT_FILE} ];then
            echo "MICRO_BATCH_SIZE,TP,PP,EP,OVERLAP_GRAD_REDUCE,OVERLAP_PARAM_GATHER,AC,AC_LAYERS,FUSED_CROSS_ENTROPY,FIRST_PP_NUM_LAYERS,LAST_PP_NUM_LAYERS,Result,TFLOPS,MFU,MemAve,MemMax,GpuUtilAve,GpuUtilMax,Log Director,Reasons for failure" > ${REPORT_FILE}
        fi
        echo "${MICRO_BATCH_SIZE},${TP},${PP},${EP},${OVERLAP_GRAD_REDUCE},${OVERLAP_PARAM_GATHER},${AC},${AC_LAYERS},${FUSED_CROSS_ENTROPY},${FIRST_PP_NUM_LAYERS},${LAST_PP_NUM_LAYERS},${result},${tflops},${mfu},${mem_ave},${mem_max},${gpu_util_ave},${gpu_util_max},${ITER_CUR},NULL" >> ${REPORT_FILE}

        echo "MICRO_BATCH_SIZE,TP,PP,EP,OVERLAP_GRAD_REDUCE,OVERLAP_PARAM_GATHER,AC,AC_LAYERS,FUSED_CROSS_ENTROPY,FIRST_PP_NUM_LAYERS,LAST_PP_NUM_LAYERS,Result,TFLOPS,MFU,MemAve,MemMax,GpuUtilAve,GpuUtilMax,Log Director,Reasons for failure" > ${REPORT_FILE_LAST}
        echo "${MICRO_BATCH_SIZE},${TP},${PP},${EP},${OVERLAP_GRAD_REDUCE},${OVERLAP_PARAM_GATHER},${AC},${AC_LAYERS},${FUSED_CROSS_ENTROPY},${FIRST_PP_NUM_LAYERS},${LAST_PP_NUM_LAYERS},${result},${tflops},${mfu},${mem_ave},${mem_max},${gpu_util_ave},${gpu_util_max},${ITER_CUR},NULL" >> ${REPORT_FILE_LAST}

        
    else
        # dense
        if [ ! -f ${REPORT_FILE} ];then
            echo "MICRO_BATCH_SIZE,TP,PP,OVERLAP_GRAD_REDUCE,OVERLAP_PARAM_GATHER,AC,AC_LAYERS,FUSED_CROSS_ENTROPY,FIRST_PP_NUM_LAYERS,LAST_PP_NUM_LAYERS,Result,TFLOPS,MFU,MemAve,MemMax,GpuUtilAve,GpuUtilMax,Log Director,Reasons for failure" > ${REPORT_FILE}
        fi
        echo "${MICRO_BATCH_SIZE},${TP},${PP},${OVERLAP_GRAD_REDUCE},${OVERLAP_PARAM_GATHER},${AC},${AC_LAYERS},${FUSED_CROSS_ENTROPY},${FIRST_PP_NUM_LAYERS},${LAST_PP_NUM_LAYERS},${result},${tflops},${mfu},${mem_ave},${mem_max},${gpu_util_ave},${gpu_util_max},${ITER_CUR},NULL" >> ${REPORT_FILE}

        echo "MICRO_BATCH_SIZE,TP,PP,OVERLAP_GRAD_REDUCE,OVERLAP_PARAM_GATHER,AC,AC_LAYERS,FUSED_CROSS_ENTROPY,FIRST_PP_NUM_LAYERS,LAST_PP_NUM_LAYERS,Result,TFLOPS,MFU,MemAve,MemMax,GpuUtilAve,GpuUtilMax,Log Director,Reasons for failure" > ${REPORT_FILE_LAST}
        echo "${MICRO_BATCH_SIZE},${TP},${PP},${OVERLAP_GRAD_REDUCE},${OVERLAP_PARAM_GATHER},${AC},${AC_LAYERS},${FUSED_CROSS_ENTROPY},${FIRST_PP_NUM_LAYERS},${LAST_PP_NUM_LAYERS},${result},${tflops},${mfu},${mem_ave},${mem_max},${gpu_util_ave},${gpu_util_max},${ITER_CUR},NULL" >> ${REPORT_FILE_LAST}
    fi


    echo "save fix parameters"
    if [ ! -f "${FIXED_PARA_FILE}" ]; then
        if [[ "$mem_limit" == "NULL" ]]; then
            mem_limit=81920
        fi
        echo "Model Type: ${MODEL_TYPE}" >> ${FIXED_PARA_FILE}
        echo "Model Size: ${MODEL_SIZE}" >> ${FIXED_PARA_FILE}
        echo "NUM_LAYERS: ${NUM_LAYERS}" >> ${FIXED_PARA_FILE}
        if [[ "${MODEL_TYPE,,}" == *"moe"* ]]; then
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
    echo "MICRO_BATCH_SIZE: ${MICRO_BATCH_SIZE}" >> ${CURRENT_PARA_FILE}
    echo "TP: ${TP}" >> ${CURRENT_PARA_FILE}
    echo "PP: ${PP}" >> ${CURRENT_PARA_FILE}
    if [[ "${MODEL_TYPE,,}" == *"moe"* ]]; then
        echo "EP:${EP}" >> ${CURRENT_PARA_FILE}
    fi
    echo "OVERLAP_GRAD_REDUCE: ${OVERLAP_GRAD_REDUCE,,}" >> ${CURRENT_PARA_FILE}
    echo "OVERLAP_PARAM_GATHER: ${OVERLAP_PARAM_GATHER,,}" >> ${CURRENT_PARA_FILE}
    echo "AC: ${AC,,}" >> ${CURRENT_PARA_FILE}
    echo "AC_LAYERS: ${AC_LAYERS}" >> ${CURRENT_PARA_FILE}
    echo "FUSED_CROSS_ENTROPY: ${FUSED_CROSS_ENTROPY,,}" >> ${CURRENT_PARA_FILE}
    echo "FIRST_PP_NUM_LAYERS: ${FIRST_PP_NUM_LAYERS}" >> ${CURRENT_PARA_FILE}
    echo "LAST_PP_NUM_LAYERS: ${LAST_PP_NUM_LAYERS}" >> ${CURRENT_PARA_FILE}

    if [ ! -f "${PARA_FILE}" ]; then
        cp ${CURRENT_PARA_FILE} ${PARA_FILE}
    fi

    echo ${result} > ${RESULT_FILE}
    echo ${result} > ${CURRENT_LOG_DIR}/monitor_result.txt


    echo "run ${train_script} ${result}"

fi