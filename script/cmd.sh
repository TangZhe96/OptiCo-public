# for single node

bash run_circle.sh gpt3 7B 21

bash run_circle.sh llama2 0.5B 21

bash run_circle.sh llama2 7B 21

bash run_circle.sh llama3 0.5B 21

bash run_circle.sh llama3 8B 21

bash run_circle.sh qwen2.5 0.5B 21

bash run_circle.sh qwen2.5 7B 21

bash run_circle.sh qwen3-moe A0.6B 21


# for multi node (change IP to the master node's IP)

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh gpt3 7B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh llama2 0.5B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh llama3 8B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen2.5 0.5B 21

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen2.5 7B 7

MASTER_ADDR=IP WORLD_SIZE=2 RANK=0 bash run_circle_multinode.sh qwen3-moe A0.6B 2
