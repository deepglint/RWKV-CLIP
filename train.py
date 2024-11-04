import os
import sys
import time
import math
import torch
import logging
import argparse
from random import choice
from loss import ClipLoss
from email.policy import default
from torch.nn import functional as F
from dali import dali_dataloader
from torch import distributed, optim
from torch.utils.tensorboard import SummaryWriter

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GlobalStep:
    def __init__(self, step: int = 0):
        self.step = int(step)

    def update(self):
        self.step += 1

class SpeedCallBack(object):
    def __init__(self, frequent, steps_total, batch_size):
        self.batch_size = batch_size
        self.frequent = frequent
        self.steps_total = steps_total
        self.loss_metric = AverageMeter()

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.time_start = time.time()
        self.init = False
        self.tic = 0

    def __call__(
            self,
            lr_scheduler: optim.lr_scheduler._LRScheduler,
            loss,
            global_step,
            scale):
        assert isinstance(loss, float)
        self.loss_metric.update(loss)

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (self.frequent * self.batch_size / (time.time() - self.tic))
                    self.tic = time.time()
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                loss_str_format = f"[{self.loss_metric.avg :.3f}]"
                self.loss_metric.reset()

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.steps_total)
                time_for_end = time_total - time_now
                lr_1 = lr_scheduler.get_last_lr()[0]
                msg = f"rank:{int(speed) :d} "
                msg += f"total:{int(speed_total) :d} "
                msg += f"lr:[{lr_1 :.8f}] "
                msg += f"step:{global_step :d} "
                msg += f"amp:{int(scale) :d} "
                msg += f"required:{time_for_end :.1f} hours "
                msg += loss_str_format

                if self.rank == 0:
                    logging.info(msg)
            else:
                self.init = True
                self.tic = time.time()

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
        
    
def get_model_RWKV_CLIP(args):
    from model import Text_RWKV, Image_RWKV, get_model
    model_image_rwkv = Image_RWKV(img_size = args.input_size,
                                patch_size= args.image_patch_size,
                                embed_dims = args.image_embed_dims,
                                hidden_rate= args.image_hidden_rate, 
                                depth=args.image_depth,
                                num_heads=args.image_num_heads,
                                output_cls_token=args.image_output_cls_token,
                                with_cls_token=args.image_with_cls_token,
                                with_cp=args.with_cp,
                                drop_path_rate=args.drop_path_rate) 
    model_text_rwkv = Text_RWKV(args)
    model = get_model(model_image_rwkv, model_text_rwkv, image_cls_token=args.image_output_cls_token)
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta1", type=float, default=0.9, help="adamw")
    parser.add_argument("--beta2", type=float, default=0.98, help="adamw")
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--gradient-acc", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--is-normlize", type=int, default=1)
    parser.add_argument("--local-loss",default=False,help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)")
    parser.add_argument("--gather-with-grad",default=False,help="enable full distributed gradient for feature gather")
    parser.add_argument("--horovod",default=False,action="store_true",help="Use horovod for distributed training.")
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--output", required=True)
    parser.add_argument('--cfg', type=str, default='', metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',help='Drop path rate (default: 0.1)')
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--train-num-samples", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument('--dropout', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument("--open-checkpoint", default="False", type=str)
    
    ################################################################################
    ################################# Image RWKV ####################################
    ################################################################################
    parser.add_argument("--input-size", default=224, type=int, help="input_image_size")
    parser.add_argument("--image-depth", default=12, type=int)
    parser.add_argument("--image-embed-dims", default=384, type=int)
    parser.add_argument("--image-patch-size", default=16, type=int)
    parser.add_argument("--image-hidden-rate", default=4, type=int)
    parser.add_argument("--image-num-heads", default=8, type=int)
    parser.add_argument("--image-output-cls-token", default="False", type=str)
    parser.add_argument("--image-with-cls-token", default="False", type=str)
    parser.add_argument("--drop-path-rate", default=0.0, type=float)

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
    
    args.text_initialization = True if args.text_initialization == "True" else  False
    args.image_output_cls_token = True if args.image_output_cls_token == "True" else False
    args.image_with_cls_token = True if args.image_output_cls_token == "True" else False
    args.with_cp = True if args.open_checkpoint == "True" else False
    
    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]
    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    assert args.image_embed_dims == args.n_embd, "Image embedding dimension must be the same as the Text embedding dimension"
    assert args.image_output_cls_token == args.image_with_cls_token, "with_cls_token must be True if set output_cls_token to True"
    
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
    if args.text_num_head != 0:
        assert args.n_embd % args.text_num_head == 0
        args.head_size = args.n_embd // args.text_num_head

    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ['RWKV_FLOAT_MODE'] = str(args.precision)
    os.environ['Image_T_max'] = str((args.input_size / args.image_patch_size)**2)
    os.environ['Text_T_max'] = str(256)
    os.environ['Image_HEAD_SIE'] = str(args.image_embed_dims // args.image_num_heads)
    
    return args

def main(args):
    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)
    if rank == 0:
        summary_writer = SummaryWriter(os.path.join(args.output, "tensorboard"))
    else:
        summary_writer = None

    start_epoch = 0
    RWKV_CLIP_model = get_model_RWKV_CLIP(args)
    train_loader = dali_dataloader(args)
    training_precision_type = torch.bfloat16  if args.precision == "bf16" else torch.float16
    
    RWKV_CLIP_model.train().cuda()
    RWKV_CLIP_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(RWKV_CLIP_model)
    RWKV_CLIP_model = torch.nn.parallel.DistributedDataParallel(
        module=RWKV_CLIP_model,
        bucket_cap_mb=32,
        find_unused_parameters=True,
        static_graph=True)

    global_step = GlobalStep()
    steps_per_epoch = args.train_num_samples // world_size // args.batch_size + 1
    steps_total = int(args.epochs * steps_per_epoch)

    contrastive_loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        use_horovod=args.horovod)

    opt = torch.optim.AdamW(
        params=[{"params": RWKV_CLIP_model.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    if args.lr_scheduler == "cosine":
        assert isinstance(args.epochs, int)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=[args.lr],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.1,
        )
    elif args.lr_scheduler == "linear":
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=opt, start_factor=1.0, end_factor=0.0,
            total_iters=int(args.epochs * steps_per_epoch))
    else:
        raise

    callback_func = SpeedCallBack(5, steps_total, args.batch_size)
    auto_scaler = torch.cuda.amp.grad_scaler.GradScaler(init_scale=128, growth_interval=200)

    for epoch in range(start_epoch, math.ceil(args.epochs)):
        for _, (img, text_token) in enumerate(train_loader):
            opt.zero_grad()
            new_text_token = []
            # text random augmentation
            for i in range(text_token.size(0)):
                choose = choice([0,1,2])
                # Raw Text
                if choose == 0:
                    new_text_token.append(text_token[i, :77].long().cuda()) 
                # Synthetic Text
                elif choose == 1:
                    new_text_token.append(text_token[i, 77*1:77*2].long().cuda()) 
                # Generated Text
                elif choose == 2:
                    new_text_token.append(text_token[i, 77*2:77*3].long().cuda()) 
            text_token = torch.stack(new_text_token, dim=0)
            img = img.cuda()
            
            with torch.cuda.amp.autocast(True, dtype=training_precision_type):
                image_embeddings, text_embeddings, logit_scale = RWKV_CLIP_model(img, text_token)
                image_embedding_norm = F.normalize(image_embeddings, dim=-1) 
                text_embedding_norm = F.normalize(text_embeddings, dim=-1) 

            loss = contrastive_loss(image_embedding_norm, text_embedding_norm, logit_scale)

            if args.precision == "bf16":
                loss.backward()
                if global_step.step % args.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(RWKV_CLIP_model.parameters(), 1)
                    opt.step()
                    opt.zero_grad()
            else:
                auto_scaler.scale(loss).backward()
                if global_step.step % args.gradient_acc == 0:
                    auto_scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(RWKV_CLIP_model.parameters(), 1)
                    auto_scaler.step(opt)
                    auto_scaler.update()
                    opt.zero_grad()

            with torch.no_grad():
                unwrap_model(RWKV_CLIP_model).logit_scale.clamp_(0, math.log(100))

            lr_scheduler.step()
            global_step.step += 1

            with torch.no_grad():
                callback_func(lr_scheduler, float(loss), global_step.step, auto_scaler.get_scale())
                if summary_writer is not None:
                    summary_writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="lr_backbone",
                                              scalar_value=lr_scheduler.get_last_lr()[0],
                                              global_step=global_step.step)
                    summary_writer.add_scalar(tag="logit_scale",
                                              scalar_value=logit_scale.item(),
                                              global_step=global_step.step)

            if global_step.step > steps_total:
                break

        train_loader.reset()
        if rank == 0: 
            torch.save(obj=RWKV_CLIP_model.state_dict(), f=os.path.join(args.output, "RWKV_CLIP_model_" + str(epoch) + ".pt"))

    if summary_writer is not None:
        summary_writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
