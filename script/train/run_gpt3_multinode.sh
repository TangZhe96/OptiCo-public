#!/bin/bash
# set -eo pipefail

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning,ignore::ResourceWarning,ignore::DeprecationWarning"
export NCCL_DEBUG="ERROR"
export TORCH_CPP_LOG_LEVEL="ERROR"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=600000 # ms
export TORCH_DISTRIBUTED_TIMEOUT=600 # s

export NCCL_CONNECT_TIMEOUT=600000 # ms
export NCCL_IB_TIMEOUT=60 
export NCCL_IB_RETRY_CNT=20 
export NCCL_SOCKET_IFNAME=ens17f0np0
export GLOO_SOCKET_IFNAME=ens17f0np0
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1

export NCCL_IB_GID_INDEX=3 
export NCCL_IB_HCA=mlx5_0 

export NCCL_SOCKET_TIMEOUT=600000 # ms
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=8

OUTPUT_BASEPATH=/workspace/code/OptiCo/output/mcore
PAI_MEGATRON_PATH=/workspace/code/OptiCo/Pai-Megatron-Patch-v0-12-3

MCORE_VERSION=${MCORE_VERSION:-"0.15.0"} # 0.15.0 / 0.14.0 / 0.13.0 / 0.12.0 / 0.6.0 
if [ ${MCORE_VERSION} = "0.15.0" ]; then
    MEGATRON_PATH=${PAI_MEGATRON_PATH}/backends/megatron/Megatron-LM-core_v0.15.0
    PRETRAIN_SCRIPT_PATH=${PAI_MEGATRON_PATH}/examples/gpt3/pretrain_gpt_core0_15_0.py
elif [ ${MCORE_VERSION} = "0.14.0" ]; then
    MEGATRON_PATH=${PAI_MEGATRON_PATH}/backends/megatron/Megatron-LM-core_v0.14.0
    PRETRAIN_SCRIPT_PATH=${PAI_MEGATRON_PATH}/examples/gpt3/pretrain_gpt_core0_14_0.py
elif [ ${MCORE_VERSION} = "0.13.0" ]; then
    MEGATRON_PATH=${PAI_MEGATRON_PATH}/backends/megatron/Megatron-LM-core_v0.13.0
    PRETRAIN_SCRIPT_PATH=${PAI_MEGATRON_PATH}/examples/gpt3/pretrain_gpt_core0_13_0.py
elif [ ${MCORE_VERSION} = "0.12.0" ]; then
    MEGATRON_PATH=${PAI_MEGATRON_PATH}/backends/megatron/Megatron-LM-core_v0.12.0
    PRETRAIN_SCRIPT_PATH=${PAI_MEGATRON_PATH}/examples/gpt3/pretrain_gpt_core0_12_0.py
elif [ ${MCORE_VERSION} = "0.6.0" ]; then
    MEGATRON_PATH=${PAI_MEGATRON_PATH}/backends/megatron/Megatron-LM-core_v0.6.0
    PRETRAIN_SCRIPT_PATH=${PAI_MEGATRON_PATH}/examples/gpt3/pretrain_gpt_core0_6_0.py
fi

export PYTHONPATH=${PAI_MEGATRON_PATH}:${MEGATRON_PATH}:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

ENV=dsw
if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    MASTER_ADDR=${MASTER_ADDR:-"localhost"}
    MASTER_PORT=${MASTER_PORT:-9987}
    NNODES=${WORLD_SIZE:-1}
    NODE_RANK=${RANK:-0}
    GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODEL_TYPE=gpt3
MODEL_SIZE=${MODEL_SIZE:-7B}
PR=${PR:-fp16}
GLOBAL_BATCH_SIZE=1024
SEQ_LENGTH=4096

if [ ${MODEL_SIZE} = "7B" ]; then
    AC=${AC:-full}
    AC_LAYERS=${AC_LAYERS:-1}
    TP=${TP:-1}
    PP=${PP:-8}
elif [ ${MODEL_SIZE} = "13B" ]; then
    AC=${AC:-none}
    AC_LAYERS=${AC_LAYERS:-1}
    TP=${TP:-1}
    PP=${PP:-1}
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY:-false}
FIRST_PP_NUM_LAYERS=${FIRST_PP_NUM_LAYERS:-none}
LAST_PP_NUM_LAYERS=${LAST_PP_NUM_LAYERS:-none}

OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE,,}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER,,}
AC=${AC,,}
FUSED_CROSS_ENTROPY=${FUSED_CROSS_ENTROPY,,}
FIRST_PP_NUM_LAYERS=${FIRST_PP_NUM_LAYERS,,}
LAST_PP_NUM_LAYERS=${LAST_PP_NUM_LAYERS,,}

comm_overlap_option=""
if [ $OVERLAP_GRAD_REDUCE = true ]; then
    comm_overlap_option+="\
                        --overlap-grad-reduce"
fi
if [ $OVERLAP_PARAM_GATHER = true ]; then
    comm_overlap_option+="\
                        --overlap-param-gather"
fi

fusion_option=""
if [ $FUSED_CROSS_ENTROPY = true ]; then
    fusion_option=" \
        --cross-entropy-loss-fusion"
fi

pp_partition_option=""
if [ $FIRST_PP_NUM_LAYERS != none ]; then
    pp_partition_option+=" \
            --decoder-first-pipeline-num-layers ${FIRST_PP_NUM_LAYERS}"
fi
if [ $LAST_PP_NUM_LAYERS != none ]; then
    pp_partition_option+=" \
            --decoder-last-pipeline-num-layers ${LAST_PP_NUM_LAYERS}"
fi

PAD_LEN=${SEQ_LENGTH}
LR=1e-5
MIN_LR=1e-6

SP=true
DO=true
FL=true

EXTRA_VOCAB_SIZE=0

TE=true
SAVE_INTERVAL=500
DATASET_PATH=/workspace/dataset/llama2_wudao/wudao_llamabpe_text_document
PRETRAIN_CHECKPOINT_PATH=/workspace/model/gpt-3
TRAIN_ITERS=5
LR_WARMUP_ITERS=0
LR_DECAY_ITERS=${TRAIN_ITERS}

if [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32

elif [ $MODEL_SIZE = 13B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5140
NUM_ATTN_HEADS=40

elif [ $MODEL_SIZE = 175B ]; then

NUM_LAYERS=96
HIDDEN_SIZE=12288
NUM_ATTN_HEADS=96

fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option+="\
                        --tp-comm-overlap"
fi


if [ $PR = fp16 ]; then
    pr_option=" \
		    --fp16 \
            --apply-query-key-layer-scaling \
            --initial-loss-scale 65536"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_option=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_option=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $DO = true ]; then
    do_option=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_option=" \
                    "
fi

if [ $FL = true ]; then
    flash_option=" \
		    --use-flash-attn"
elif [ $FL = false ]; then
    flash_option=" \
                    "
fi

if [ $TE = true ]; then
    te_option=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_option=" \
        --transformer-impl local"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
                    "
fi


dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 99,1,0 \
    --dataset MMAP"

NAME="pretrain-mcore-${MODEL_TYPE}-${MODEL_SIZE}-mbs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LENGTH}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

if [[ -z ${LOG_FILE} ]];then
    LOG_DIR=${OUTPUT_BASEPATH}/${current_time}_log_${NAME}
    mkdir -p ${LOG_DIR}
    LOG_NAME="${NODE_RANK}.txt"
    LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi


if [ $AC = full ]; then
    activation_checkpoint_option=" \
		    --recompute-method uniform \
            --recompute-num-layers ${AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_option=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_option=" \
                    "
elif [ $AC = offload ]; then
    activation_checkpoint_option=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

megatron_option="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings ${SEQ_LENGTH} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 0 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 1 \
        --seed 1234 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ${PRETRAIN_CHECKPOINT_PATH}/vocab.json \
        --merge-file ${PRETRAIN_CHECKPOINT_PATH}/merges.txt \
        --swiglu \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --use-mcore-models \
        "
        # --use-mcore-models \
run_cmd="torchrun $DISTRIBUTED_ARGS ${PRETRAIN_SCRIPT_PATH}
 ${megatron_option} ${dataset_option} ${pr_option} ${te_option} ${activation_checkpoint_option} ${do_option} ${flash_option} ${sp_option} ${comm_overlap_option} ${fusion_option} ${pp_partition_option}"

echo ${run_cmd} >> ${LOG_FILE}
echo "AC is ${AC}" >> ${LOG_FILE}
echo "AC_LAYERS is ${AC_LAYERS}" >> ${LOG_FILE}

# check NUM_LAYERS
if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "Error: the num layers per pp rank must be a multiple of the recompute layers."  >> ${LOG_FILE}
        exit -1
    fi
fi

FIRST_PP_LAYERS=$((FIRST_PP_NUM_LAYERS == "none" ? 0 : FIRST_PP_NUM_LAYERS))
LAST_PP_LAYERS=$((LAST_PP_NUM_LAYERS == "none" ? 0 : LAST_PP_NUM_LAYERS))

if [ "$PP" -eq 1 ] && ([ "$FIRST_PP_NUM_LAYERS" != "none" ] || [ "$LAST_PP_NUM_LAYERS" != "none" ]); then
    echo "Error: When PP=1, FIRST_PP_NUM_LAYERS and LAST_PP_NUM_LAYERS should be none, but FIRST_PP_NUM_LAYERS=${FIRST_PP_NUM_LAYERS} LAST_PP_NUM_LAYERS=${LAST_PP_NUM_LAYERS}"  >> ${LOG_FILE}
    exit -1
fi
if [ "$PP" -gt 1 ] && ([ "$FIRST_PP_NUM_LAYERS" != "none" ] || [ "$LAST_PP_NUM_LAYERS" != "none" ]); then
    
    # Validate conditions
    if [ "$PP" -eq 2 ]; then
        # When PP=2, sum of first and last layers must equal total layers
        if [ $((FIRST_PP_LAYERS + LAST_PP_LAYERS)) -ne "$NUM_LAYERS" ]; then
            echo "Error: When PP=2, FIRST_PP_NUM_LAYERS + LAST_PP_NUM_LAYERS must equal NUM_LAYERS"  >> ${LOG_FILE}
            exit -1
        fi
    else
        # When PP>2, (NUM_LAYERS - FIRST - LAST) must be divisible by (PP-2)
        middle_layers=$((NUM_LAYERS - FIRST_PP_LAYERS - LAST_PP_LAYERS))
        
        # Check if middle_layers is positive
        if [ "$middle_layers" -le 0 ]; then
            echo "Error: Middle layers($middle_layers) must be greater than 0"  >> ${LOG_FILE}
            exit -1
        fi
        
        if [ $((middle_layers % (PP - 2))) -ne 0 ]; then
            echo "Error: Middle layers($middle_layers) must be divisible by (PP-2)=$((PP - 2))" >> ${LOG_FILE}
            exit -1
        fi
    fi
    
fi

eval ${run_cmd} 2>&1 | stdbuf -o0 grep -v -E "Warning:|FutureWarning|UserWarning|RuntimeWarning|ResourceWarning|DeprecationWarning" | tee -a ${LOG_FILE}

set +x
