#!/bin/bash
# set -eo pipefail

export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::RuntimeWarning,ignore::ResourceWarning,ignore::DeprecationWarning"
export NCCL_DEBUG="ERROR"
export TORCH_CPP_LOG_LEVEL="ERROR"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=60000
export TORCH_DISTRIBUTED_TIMEOUT=60

OUTPUT_BASEPATH=/workspace/code/OptiCo/output/mcore
MEGATRON_PATH=/workspace/code/OptiCo/Pai-Megatron-Patch-v0-12-3
# MEGATRON_PATH=/workspace/code/OptiCo/Pai-Megatron-Patch-2b201af
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/backends/megatron/Megatron-LM-250624:$PYTHONPATH
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

MODEL_TYPE="qwen3-moe"
MODEL_SIZE=${MODEL_SIZE:-A0.6B} # 6B
PR=${PR:-bf16}
GLOBAL_BATCH_SIZE=1024
SEQ_LENGTH=4096

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-1}
OVERLAP_GRAD_REDUCE=${OVERLAP_GRAD_REDUCE:-false}
OVERLAP_PARAM_GATHER=${OVERLAP_PARAM_GATHER:-false}
AC=${AC:-none}
AC_LAYERS=${AC_LAYERS:-1}
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

CP=1
ETP=1
SP=true
DO=true
FL=true
SFT=false
TE=true

OPTIMIZER_OFFLOAD=false
DATASET_PATH=/workspace/dataset/qwen3_datasets/mmap_qwen3_datasets_text_document
VALID_DATASET_PATH=None
PRETRAIN_CHECKPOINT_PATH=/workspace/model/Qwen3-mini-moe
TRAIN_ITERS=5
LR_WARMUP_ITERS=0
LR_DECAY_ITERS=${TRAIN_ITERS}

SAVE_INTERVAL=100
SAVE_CKPT=false
LOAD_CKPT=false
LOAD_CKPT_PATH="/workspace/code/OptiCo/output/mcore/checkpoint/pretrain-mcore-qwen3-moe-A0.6B-pr-bf16-tp-1-pp-1-cp-1-ac-none-do-true-sp-true-ti-5-wi-0"
ckpt_opt=""


if [ $MODEL_SIZE = A0.6B ]; then
    HIDDEN_SIZE=1024
    NUM_ATTN_HEADS=16
    NUM_LAYERS=28
    INTERMEDIATE_SIZE=3072
    MOE_INTERMEDIATE_SIZE=384
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=8
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_option=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*28)' \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_option=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"


elif [ $MODEL_SIZE = A3B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=32
    NUM_LAYERS=48
    INTERMEDIATE_SIZE=6144
    MOE_INTERMEDIATE_SIZE=768
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_option=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*48)' \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
    gqa_option=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
elif [ $MODEL_SIZE = A22B ]; then
    HIDDEN_SIZE=4096
    NUM_ATTN_HEADS=64
    NUM_LAYERS=94
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6


    moe_option=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-layer-freq '([1]*94)' \
        --moe-router-pre-softmax
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_option=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
fi


if [ -z ${MP_VP} ]; then
    vp_option=""
else
    vp_option=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option+="\
        --tp-comm-overlap"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
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

if [ $PR = fp16 ]; then
    pr_option=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_option=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_option=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-recipe blockwise"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
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
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
        --attention-backend flash
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    flash_option=" \
                    "
fi

if [ $TE = true ]; then
    te_option=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_option=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_option=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

if [ $SFT = true ]; then
    PREFIX="finetune-mcore-${MODEL_TYPE}-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LENGTH}"
    sft_option=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    PREFIX="pretrain-mcore-${MODEL_TYPE}-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LENGTH}"
    sft_option=" \
        --train-mode pretrain"
fi

if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi
if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT"
else 
    dataset_option=" \
        --data-path ${DATASET_PATH} \
        --split 99,1,0 \
        --dataset MMAP"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_option=" \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader
    "
else
    packing_option=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

if [ $SAVE_CKPT = true ]; then
    SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
    mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

    ckpt_opt="\
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --save-interval ${SAVE_INTERVAL} \
    "
fi

if [ $LOAD_CKPT = true ]; then
    ckpt_opt="\
        --load ${LOAD_CKPT_PATH} \
    "
fi


if [[ -z ${LOG_FILE} ]];then
    LOG_DIR=${OUTPUT_BASEPATH}/${current_time}_log_${NAME}
    mkdir -p ${LOG_DIR}
    LOG_NAME="${NODE_RANK}.txt"
    LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi

megatron_option="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LENGTH} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 0 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 1 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen3Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --ckpt-format torch_dist \
        --qk-layernorm \
        --kv-channels 128 \
        --te-rng-tracker \
        "
        # --external-cuda-graph \
        # --cuda-graph-scope attn


run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/examples/qwen3/pretrain_qwen.py
 ${megatron_option} ${dataset_option} ${pr_option} ${load_option} ${te_option} ${activation_checkpoint_option} ${do_option} ${flash_option} ${sp_option} ${gqa_option} ${moe_option} ${offload_option} ${sft_option} ${tie_option} ${vp_option} ${ckpt_opt} ${packing_option} ${comm_overlap_option} ${fusion_option} ${attn_backend_option} ${pp_partition_option}"

echo ${run_cmd} >> ${LOG_FILE}
echo "AC is ${AC}" >> ${LOG_FILE}
echo "AC_LAYERS is ${AC_LAYERS}" >> ${LOG_FILE}

# check NUM_LAYERS
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
