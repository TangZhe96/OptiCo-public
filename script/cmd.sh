# for single node

bash run_circle.sh gpt3 7B 21

bash run_circle.sh llama2 0.5B 21

bash run_circle.sh llama2 7B 21

bash run_circle.sh llama3 0.5B 21

bash run_circle.sh llama3 8B 21

bash run_circle.sh qwen2.5 0.5B 21

bash run_circle.sh qwen2.5 7B 21

bash run_circle.sh qwen3-moe A0.6B 21

# V5000 DeepSeek-V2 / Llama3 (training via ../Meta-OptiCo/nhmegatron). Prefer Meta-OptiCo/cmd:
#   bash ../Meta-OptiCo/cmd/cmd_v5000_16b_public.sh
#   bash ../Meta-OptiCo/cmd/cmd_v5000_236b_public.sh
#   bash ../Meta-OptiCo/cmd/cmd_v5000_llama3_70b_public.sh
PLATFORM=V5000 bash run_circle.sh deepseek_v2 A2.4B 20
PLATFORM=V5000 bash run_circle.sh deepseek_v2 A21B 20
PLATFORM=V5000 bash run_circle.sh llama3 70B 20

# W64 021 MoE (training via ../Meta-OptiCo/nhmegatron). Prefer Meta-OptiCo/cmd:
#   bash ../Meta-OptiCo/cmd/cmd_w64_021_32b_public.sh
#   bash ../Meta-OptiCo/cmd/cmd_w64_021_236b_public.sh
PLATFORM=W64 bash run_circle.sh 021 32B 20
PLATFORM=W64 bash run_circle.sh 021 236B 20


# for multi node (change IP to the master node's IP)

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh gpt3 7B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh llama2 0.5B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh llama3 8B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen2.5 0.5B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen2.5 7B 7

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen3-moe A0.6B 2
