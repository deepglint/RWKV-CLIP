import torch
import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict["data"]
        tensor_label: torch.Tensor = data_dict["label"]
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter.__len__()

    def reset(self):
        self.iter.reset()


def dali_dataloader(args, ):
    input_filename = args.train_data
    assert input_filename

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rec_file = f"{input_filename}.rec"
    idx_file = f"{input_filename}.idx"
    
    pipe = Pipeline(
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=local_rank,
        prefetch_queue_depth=3,
        seed=rank + 1467,
    )

    is_training = True
    device_memory_padding = 211025920
    host_memory_padding = 140544512

    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file,
            index_path=idx_file,
            initial_fill=16384,
            num_shards=world_size,
            shard_id=rank,
            random_shuffle=True,
            pad_last_batch=False,
            name="train",
        )
        if is_training:
            images = fn.decoders.image_random_crop(
                jpegs,
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.7, 1.0],
                num_attempts=100,
            )
            images = fn.resize(
                images,
                device="gpu",
                resize_x=args.input_size,
                resize_y=args.input_size,
                interp_type=types.INTERP_LINEAR,
            )
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(
                images,
                device="gpu",
                size=int(256 / 224 * args.input_size),
                mode="not_smaller",
                interp_type=types.INTERP_LINEAR,
            )
            mirror = False
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(args.input_size, args.input_size),
            mean=[x * 255 for x in mean],
            std=[x * 255 for x in std],
            mirror=mirror)
        pipe.set_outputs(images,labels)
    pipe.build()

    dataloader = DALIWarper(
        DALIClassificationIterator(pipelines=[pipe], reader_name="train"),
    )
    return dataloader
