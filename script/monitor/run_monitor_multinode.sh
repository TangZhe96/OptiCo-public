#!/bin/bash
# Multi-node wrapper: same monitor as single-node, Pai-Megatron *_multinode.sh.
export OPTICO_TRAIN_MULTINODE=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_monitor.sh"
