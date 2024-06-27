from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Food101
from .utils import dataset_root


root = dataset_root
num_example_train_val = 75750
num_example_test = 25250
num_classes = 101


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = Food101(root, download=True, split='train', transform=transform)
    return (dataset_train_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = Food101(root, download=True, split='test', transform=transform)
    return (dataset_test, None)
