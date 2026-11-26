#!/bin/bash
current_date=$(date +%m%d)

sudo docker run --net host --gpus all \
    --name optico-${current_date} \
    --privileged --cap-add=SYS_PTRACE \
    --shm-size=80g \
    -v /mnt/code/optico:/workspace/code/OptiCo \
    -v /mnt/dataset:/workspace/dataset \
    -v /mnt/model:/workspace/model \
    -w /workspace/optico \
    -it nvcr.io/nvidia/pytorch:24.11-py3
