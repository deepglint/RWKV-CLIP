from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import StanfordCars
from .utils import dataset_root


root = dataset_root
num_example_train_val = 8144
num_example_test = 8041
num_classes = 196


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = StanfordCars(
        root, download=False, split='train', transform=transform)
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = StanfordCars(
        root, download=False, split='test', transform=transform)
    return (dataset_test, None)
