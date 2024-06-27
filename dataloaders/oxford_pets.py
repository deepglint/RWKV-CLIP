from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from .utils import dataset_root


root = dataset_root
num_example_train_val = 3680
num_example_test = 3699
num_classes = 37
mean_per_class = True


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = OxfordIIITPet(root, download=True, split='trainval',transform=transform)
    return (dataset_train_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = OxfordIIITPet(root, download=True, split='test', transform=transform)
    return (dataset_test, None)

