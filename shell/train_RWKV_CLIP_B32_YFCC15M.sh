######################### Image Setting #########################
input_size=224
image_depth=12
image_embed_dims=640
image_patch_size=32
image_hidden_rate=5
image_num_heads=8
drop_path_rate=0.3

######################### Text Setting #########################
ctx_len=77
vocab_size=49408
head_size=64
head_size_divisor=8
n_embd=640
n_layer=6
text_initialization=True

######################### Others #########################
ip_list=(YOUR_ADDRESS) # e.g. one node ("1.1.1.1"); multi node ("1,1,1,1" "2,2,2,2")
lr=0.001
opt=adamw
weight_decay=0.2
train_num_samples=15061515
epochs=32
batch_size=512
precision=bf16
open_checkpoint=False
traindata=TRAINING_DATA_PATH
output=OUTPUT_PATH

for((node_rank=0;node_rank<${#ip_list[*]};node_rank++));
do
  ssh root@${ip_list[node_rank]} "cd `pwd`;PATH=$PATH \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NCCL_ALGO=Ring \
  NCCL_SOCKET_IFNAME=eth0 \
  NCCL_SOCKET_NTHREADS=8 \
  NCCL_NSOCKS_PERTHREAD=2 \
  torchrun --nproc_per_node 8 \
    --nnodes=${#ip_list[*]} \
    --node-rank=$node_rank \
    --master-addr=${ip_list[0]} \
    --master-port=23760  train.py \
    --drop-path-rate $drop_path_rate \
    --image-num-heads $image_num_heads \
    --input-size $input_size \
    --dropout $dropout \
    --precision $precision \
    --image-depth $image_depth \
    --image-embed-dims $image_embed_dims \
    --image-patch-size $image_patch_size \
    --image-hidden-rate $image_hidden_rate \
    --ctx-len $ctx_len \
    --vocab-size $vocab_size \
    --head-size $head_size \
    --head-size-divisor $head_size_divisor \
    --n-embd $n_embd \
    --n-layer $n_layer \
    --text-initialization $text_initialization \
    --batch-size $batch_size \
    --epochs $epochs \
    --lr $lr \
    --optimizer $opt \
    --output $output \
    --train-data $traindata \
    --train-num-samples $train_num_samples \
    --weight-decay $weight_decay" &
done
