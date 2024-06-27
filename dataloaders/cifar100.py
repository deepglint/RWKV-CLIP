from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from .utils import dataset_root


root = dataset_root
num_example_train_val = 50000
num_example_test = 10000
num_classes = 100


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = CIFAR100(root, download=False, train=True, transform=transform)
    return (dataset_train_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = CIFAR100(root, download=False, train=False, transform=transform)
    return (dataset_test, None)
