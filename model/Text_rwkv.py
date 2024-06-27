# Copyright (c) Shanghai AI Lab. All rights reserved.
import os, math, gc
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn import functional as F


from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])
T_MAX = int(float(os.environ['Text_T_max']))

def __nop(ob):
    return ob

MyModule = nn.Module
MyFunction = __nop


########################################################################################################
# CUDA Kernel
########################################################################################################
wkv6_text_cuda = load(name="wkv6_text_birectional",
                sources=["model/cuda_text/wkv6_text_op.cpp", 
                        "model/cuda_text/wkv6_text_cuda.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                "-O3", "-Xptxas -O3", 
                "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", 
                f"-D_T_={T_MAX}"])

class WKV_6_bidirectional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_text_cuda.forward(B, T, C, H, r.float(), k.float(), v.float(), ew, u.float(), y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)
            
            wkv6_text_cuda.backward(B, T, C, H, r.float(), k.float(), v.float(), ew, u.float(), gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6_bidirectional(B, T, C, H, r, k, v, w, u):
    return WKV_6_bidirectional.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################
# Shift Method
########################################################################################################
def bidirectional_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE):
    B, N, C = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    dim_mid = int(head_dim/2)
    input = input.transpose(1, 2).reshape(B, -1, head_dim, N) 
    output = torch.zeros_like(input) 
    output[:, :, :dim_mid, shift_pixel:N] = input[:, :, :dim_mid, 0:N-shift_pixel]
    output[:, :, dim_mid:, 0:N-shift_pixel] = input[:, :, dim_mid:, shift_pixel:N]
    output = output.reshape(B, C, N).transpose(1, 2)
    return output


########################################################################################################
# Time Mix
########################################################################################################
class RWKV_Tmix_V6(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        
        self.RUN_CUDA_RWKV6_choose = RUN_CUDA_RWKV6_bidirectional

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att, dtype=torch.bfloat16)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att, dtype=torch.bfloat16)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = bidirectional_shift_multihead
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        # print(x)
        # print(x.dtype)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        def _inner_forward(x):
            B, T, C = x.size()
            H = self.n_head

            r, k, v, g, w = self.jit_func(x)
            x = self.RUN_CUDA_RWKV6_choose(B, T, C, H, r, k, v, w, u=self.time_faaaa)

            return self.jit_func_2(x, g)
        if self.args.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


########################################################################################################
# Channel Mix
########################################################################################################
class RWKV_CMix_V6(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = bidirectional_shift_multihead

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        def _inner_forward(x):
            xx = self.time_shift(x) - x
            xk = x + xx * self.time_maa_k
            xr = x + xx * self.time_maa_r

            k = self.key(xk)
            k = torch.relu(k) ** 2
            kv = self.value(k)
            return torch.sigmoid(self.receptance(xr)) * kv
        if self.args.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


########################################################################################################
# Block
########################################################################################################
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.pos_emb,1,args.n_embd)))

        self.att = RWKV_Tmix_V6(args, layer_id)
        self.ffn = RWKV_CMix_V6(args, layer_id)
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        # if args.dropout > 0:
        self.drop0 = nn.Dropout(p = args.dropout)
        self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, x_emb=None):
        def _inner_forward(x):
            args = self.args
            B, T, C = x.size()
            if self.layer_id == 0:
                x = self.ln0(x)
                if args.pos_emb > 0:
                    pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                    x = x + pos_emb

            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

            if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
                xx = self.tiny_ln(x)
                q = self.tiny_q(xx)[:, :T, :]
                k = self.tiny_k(xx)[:, :T, :]
                c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
                c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
                x = x + c @ self.tiny_v(x_emb)
            return x
        if self.args.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


########################################################################################################
# Init Method
########################################################################################################
def generate_init_weight_V6(model):
        m = {}
        n_params = 0
        for n in model.state_dict():
            # print(n)
            p = model.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / model.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1e-4
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])

                    zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                    for kk in [".att.key."]:
                        if kk in n:
                            scale = 0.1
                    for kk in [".ffn.key."]:
                        if kk in n:
                            scale = 0.5
                    for kk in [".att.gate."]:
                        if kk in n:
                            scale = 0.1

                m[n] = torch.empty((shape[0], shape[1]), device="cuda")

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain= gain * scale)

            # m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()
            
        gc.collect()
        torch.cuda.empty_cache()
        return m


class Text_RWKV(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        if not hasattr(args, 'tiny_att_layer'):
            args.tiny_att_layer = -1
        if not hasattr(args, 'tiny_att_dim'):
            args.tiny_att_dim = -1

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
        
        if args.text_initialization:  
            self.load_state_dict(generate_init_weight_V6(self), strict=True)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
        
        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.blocks:
                x = block(x, x_emb)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.ln_out(x)
        
        return x
