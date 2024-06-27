from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import SUN397
from .utils import dataset_root

root = dataset_root
num_example_train = 19850
num_example_test = 19850
num_example_others = 69054
num_classes = 397



def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = SUN397(root, transform=transform)
    dataset_train, dataset_test, others = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test,
                 num_example_others,],
        generator=torch.Generator().manual_seed(seed + hash("sun397") % 2048))
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = SUN397(root, transform=transform)
    dataset_train, dataset_test, others = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test,
                 num_example_others,],
        generator=torch.Generator().manual_seed(seed + hash("sun397") % 2048))
    return (dataset_test, None)
