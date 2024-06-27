import os
import torch
import argparse
import numpy as np
from torch import nn
from dataloaders import coco, flickr30k
import random

dataset_dict = {'coco': coco,
                'flickr': flickr30k}

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
def load_model_weight(model, model_weight)
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        k_removed = k
        if "module." in k_removed:
            k_removed = k.split("module.")[-1]
        if '_orig_mod.' in k_removed:
            k_removed = k_removed.split('_orig_mod.')[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k_removed] = value
    model.load_state_dict(state_dict_removed, strict=True)
    return model        


def main(args):
    setup_seed(1024, True)
    tokenize_model = tokenize
    model_image_rwkv = Image_RWKV(img_size = args.input_size,
                            patch_size= args.image_patch_size,
                            embed_dims = args.image_embed_dims, 
                            hidden_rate= args.image_hidden_rate, 
                            depth=args.image_depth,
                            num_heads=args.image_num_heads,
                            output_cls_token=args.image_output_cls_token,
                            with_cls_token=args.image_with_cls_token)
            
    model_text_rwkv = Text_RWKV(args)
    model = get_model(model_image_rwkv, model_text_rwkv, image_cls_token=args.image_output_cls_token)
    model = load_model_weight(model, args.model_weight)
    model.eval()
    model.cuda()

    transform = get_transform(args.input_size)
    dataset_module = dataset_dict[args.dataset]
    assert hasattr(dataset_module, "get_loader_image")
    assert hasattr(dataset_module, "get_loader_text")

    kwargs_text = {
        "batch_size": args.batch_size,
        "preprocess": transform,
        "tokenize":tokenize_model}
    kwargs_image = {
        "batch_size": args.batch_size,
        "preprocess": transform}
    
    text_loader = dataset_module.get_loader_text(**kwargs_text)
    text_features = get_text_feature(model, text_loader, args)

    image_loader, txt2img, img2txt = dataset_module.get_loader_image(**kwargs_image)
    image_features = get_image_feature(model, image_loader, args)

    ## unified image & text dtype
    text_features = torch.tensor(text_features, dtype=torch.float32)
    image_features = torch.tensor(image_features, dtype=torch.float32)
    
    similarity_scores = image_features.cuda() @ text_features.cuda().t()
    similarity_scores = similarity_scores
    t2i_dict, i2t_dict = compute_retrieval(similarity_scores, txt2img, img2txt)
    print('Text retrieval', i2t_dict)
    print('Image retrieval', t2i_dict)


def compute_retrieval(similarity_scores, txt2img, img2txt):
    # comput text -> image
    t2i_similarity_score = similarity_scores.t()
    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])

    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending= True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]
        print('Evaluating batch {}/{}, {}'.format(index, t2i_similarity_score.shape[0], t2i_ranks[index]), end = "\r")

    # Compute metrics
    tr1 = 100.0 * len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    tr5 = 100.0 * len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    tr10 = 100.0 * len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks)

    t2i_report_dict = {"r1": tr1, "r5": tr5, "r10": tr10}

    #comput image -> text
    i2t_similarity_score = similarity_scores
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    for index, score in enumerate(i2t_similarity_score):
        print('Evaluating batch {}/{}'.format(index, i2t_similarity_score.shape[0]), end = "\r")
        inds = torch.argsort(score, descending= True)
        # Score
        rank = 1e10
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    ir5 = 100.0 * len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    ir10 = 100.0 * len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
    i2t_report_dict = {"r1": ir1, "r5": ir5, "r10": ir10}
    return t2i_report_dict, i2t_report_dict

def get_image_feature(model, data_loader, args):
    image_features = []
    for batch_idx, batch in enumerate(data_loader):
        print('Evaluating batch {}/{}'.format(batch_idx, len(data_loader)), end = "\r")
        images, _ = batch
        images = images.cuda()
        image_embedding = WarperCLIP_V_T_RWKV_method(model, images)
        image_features.append(image_embedding.detach().cpu())
    print('final batch image embedding mean:', torch.mean(image_embedding))
    image_features = torch.cat(image_features, 0)
    
    print('Done image feature extract.')
    print(image_features.shape)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

def get_text_feature(model, data_loader, args):
    text_features = []
    for batch_idx, batch in enumerate(data_loader):
        print('Evaluating batch {}/{}'.format(batch_idx, len(data_loader)), end = "\r")
        text = batch.squeeze()
        text = text.cuda()
        text_embedding = WarperCLIP_V_T_RWKV_text_change_head(model, text) # 
        text_features.append(text_embedding.detach().cpu())
    print('final batch text tokens:', batch[-1])
    text_features = torch.cat(text_features, 0)
    print('Done text feature extract.')
    print(text_features.shape)

    # normalized features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

def get_transform(image_size):
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    preprocess = image_transform(image_size, is_train=False, mean=image_mean, std=image_std)
    return preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zero-shot Retrieval")
    parser.add_argument("--batch-size", default=128, type=int,
                        help="Name of the dataset to use.")
    parser.add_argument("--dataset", default="coco", type=str)#flickr&coco
    parser.add_argument("--model-type", default="rwkv_clip", type=str) #vision_rwkv/
    parser.add_argument("--model-name", default="ViT-B-32-384")
    parser.add_argument("--model-weight", default= "WEIGHT_PATH")
    parser.add_argument("--output-dir", default="OUTPUT_PATH", type=str, )
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument('--dropout', type=float, default=0.0, metavar='PCT',help='Dropout rate (default: 0.)')
    
    ################################################################################
    ################################# Image RWKV ####################################
    ################################################################################
    parser.add_argument("--input-size", default=224, type=int, help="input_image_size")
    parser.add_argument("--image-depth", default=12, type=int)
    parser.add_argument("--image-embed-dims", default=384, type=int)
    parser.add_argument("--image-patch-size", default=16, type=int)
    parser.add_argument("--image-hidden-rate", default=4, type=int)
    parser.add_argument("--image-num-heads", default=8, type=int)
    parser.add_argument("--image-output-cls_token", default="False", type=str)
    parser.add_argument("--image-with-cls-token", default="False", type=str)

    ################################################################################
    ################################# Text RWKV ####################################
    ################################################################################
    parser.add_argument("--data-type", default="utf-8", type=str)
    parser.add_argument("--ctx-len", default=77, type=int, help="")
    parser.add_argument("--vocab-size", default=49408, type=int, help="Vocabular number")
    parser.add_argument("--text-initialization", default="True", type=str)
    parser.add_argument("--head-size", default=8, type=int) 
    parser.add_argument("--text-num-head", default=0, type=int)
    parser.add_argument("--pos-emb", default=0, type=int)
    parser.add_argument("--head-size-divisor", default=8, type=int)
    parser.add_argument("--n-layer", default=12, type=int)
    parser.add_argument("--n-embd", default=384, type=int)
    parser.add_argument("--dim-att", default=0, type=int)
    parser.add_argument("--dim-ffn", default=0, type=int)
    parser.add_argument("--head-qk", default=0, type=int)
    parser.add_argument("--tiny-att-dim", default=0, type=int) 
    parser.add_argument("--tiny-att-layer", default=-999, type=int) 
    args = parser.parse_args()

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]
    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    assert args.image_embed_dims == args.n_embd
    if args.text_initialization == "True":
        args.text_initialization = True
    else:
        args.text_initialization = False

    if args.image_output_cls_token == "True":
        args.image_output_cls_token = True
        args.image_with_cls_token = True
    else:
        args.image_output_cls_token = False
        args.image_with_cls_token = False
        
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
    if args.text_num_head != 0:
        assert args.n_embd % args.text_num_head == 0, 
        args.head_size = args.n_embd // args.text_num_head
    args.with_cp = False
    
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ['RWKV_FLOAT_MODE'] = str(args.precision)
    os.environ['Image_T_max'] = str((args.input_size/args.image_patch_size)**2)
    os.environ['Text_T_max'] = str(256)
    os.environ['Image_HEAD_SIE'] = str(args.image_embed_dims // args.image_num_heads)
    
    from model import Text_RWKV, Image_RWKV, get_model, tokenize, image_transform
    from model.utils import WarperCLIP_V_T_RWKV_text_change_head, WarperCLIP_V_T_RWKV_method
    main(args)
