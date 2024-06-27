from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import Caltech101
from .utils import dataset_root


root = dataset_root
num_example_train = 3000
num_example_test = 5677
num_classes = 101
mean_per_class = True

def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = Caltech101(root, download=False, transform=transform)
    dataset_train, dataset_test = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = Caltech101(root, download=False, transform=transform)
    dataset_train, dataset_test = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_test, None)
