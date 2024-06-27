import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.
    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.
    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

##################################################################################
######################################### Image part #############################
##################################################################################    
def WarperCLIP_V_T_RWKV_method(model, image):
    if not model.image_cls_token:
        image_embedding = model.image_model(image) 
        image_embedding = model.avg_layer(image_embedding)[-1]
    else:
        image_embedding = model.image_model(image)[0][-1]
    return image_embedding

class WarperCLIP_V_T_RWKV(nn.Module):
    def __init__(self, model_clip):
        super().__init__()
        self.model_clip = model_clip

    def forward(self, image):
        return WarperCLIP_V_T_RWKV_method(self.model_clip, image)
    

##################################################################################
######################################### Text part #############################
##################################################################################
def WarperCLIP_V_T_RWKV_text_change_head(model, text):
    text_embedding = model.text_model(text)
    text_embedding = F.normalize(text_embedding, dim=-1)
    text_embedding = model.text_head(text_embedding.transpose(1,2)).squeeze(dim=-1)
    return text_embedding

##################################################################################
######################################### Whole Model
##################################################################################
class get_model(nn.Module):
    def __init__(self, model_image_rwkv, model_text_rwkv, image_cls_token=False) -> None:
        super().__init__()
        self.text_model = model_text_rwkv
        self.image_model = model_image_rwkv
        self.image_cls_token = image_cls_token
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.avg_layer = GlobalAveragePooling()
        self.text_head = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, image, text):
        ################## Image ##################
        image_embedding = WarperCLIP_V_T_RWKV_method(self, image)
        ################## Text ##################
        text_embedding = WarperCLIP_V_T_RWKV_text_change_head(self, text)
        return image_embedding, text_embedding, self.logit_scale.exp()