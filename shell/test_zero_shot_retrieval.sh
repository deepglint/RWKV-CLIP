######################### Image Setting #########################
input_size=224
image_depth=12
image_embed_dims=640
image_patch_size=32
image_hidden_rate=5
image_num_heads=8

######################### Text Setting #########################
ctx_len=77
vocab_size=49408
head_size=64
head_size_divisor=8
n_embd=640
n_layer=6
text_initialization=True

######################### Others #########################
model_name=ViT-B-32-384
batch_size=120
dropout=0.0
precision=bf16

######################### Eval Settings #########################
eval_file=MODEL_WEIGHT_PATH                      
model_type=V_T_rwkv                    
output_file=OUTPUT_PATH

Retrieval=True
node_index=1
retrieval_output_dir=${output_file}/retrieval.txt

if [ ! -d "$output_file" ];then
    mkdir $output_file
fi

model_weight=${eval_file}
output_file=${output_file}/eval
if [ ! -d "$output_file" ];then
    mkdir $output_file
fi



################################################################## cross-modal retrieval ############################################################
if [ ${Retrieval} == True ];then
    echo "############################################################ Retrieval Start ############################################################"
    CUDA_VISIBLE_DEVICES=$node_index python text_image_retrieval.py\
        --image-num-heads $image_num_heads \
        --batch-size $batch_size \
        --model-type $model_type \
        --model-name $model_name \
        --model-weight $model_weight \
        --dataset coco\
        --output-dir $retrieval_output_dir \
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
        --text-initialization $text_initialization

    CUDA_VISIBLE_DEVICES=$node_index python text_image_retrieval.py\
        --image-num-heads $image_num_heads \
        --batch-size $batch_size \
        --model-type $model_type \
        --model-name $model_name \
        --model-weight $model_weight \
        --dataset flickr \
        --output-dir $retrieval_output_dir \
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
        --text-initialization $text_initialization
fi