# NCCL_SOCKET_IFNAME=eno2 \
###
 # @Author: rxin rxin@connect.ust.hk
 # @Date: 2023-02-14 20:49:05
 # @LastEditors: rxin rxin@connect.ust.hk
 # @LastEditTime: 2023-02-15 21:16:22
 # @FilePath: /DIPP/start_train_test.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# OMP_NUM_THREADS=24 \
# torchrun \
# --nproc_per_node=1 \
# --nnodes=1 \
# --node_rank=0 \
python \
train.py \
--name DIPP_test \
--train_set /workspace/DIPP/dataset/test_tmp \
--valid_set /workspace/DIPP/dataset/test_tmp \
--use_planning \
--seed 3407 \
--num_workers 24 \
--pretrain_epochs 5 \
--train_epochs 20 \
--batch_size 16 \
--learning_rate 1.5e-4 \
--device cuda \
# --ckpt training_log/RISK/model_51_9.9982.pth
# --local_rank 0 \
# --world_size 1
# --ckpt training_log/DIPP/model_1_0.6659.pth \
# --config config.yaml \

# --rdzv_id=10086 \
# --rdzv_backend=c10d \
# --max_restarts=3 \
# --rdzv_endpoint=$MASTER_ADDR:29400
# --master_addr="192.168.200.200" \
# --master_port="7891" \
# --master_addr="10.30.10.113" \
# --master_port="7891" \