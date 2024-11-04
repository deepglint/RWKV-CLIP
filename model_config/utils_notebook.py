import os
import json

##################################################################################
######################################### Inference ##############################
##################################################################################
def load_model_configs(config_path):
    class Args:
        pass

    args = Args()
    with open(config_path, 'rt') as f:
        params = json.load(f)
        for key, value in params.items():
            setattr(args, key, value)
            
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ['RWKV_FLOAT_MODE'] = str(args.precision)
    os.environ['Image_T_max'] = str((args.input_size/args.image_patch_size)**2)
    os.environ['Text_T_max'] = str(256)
    os.environ['Image_HEAD_SIE'] = str(args.image_embed_dims // args.image_num_heads)  
    
    return args