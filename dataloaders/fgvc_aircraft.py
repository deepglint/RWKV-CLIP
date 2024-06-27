from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FGVCAircraft
from .utils import dataset_root


root = dataset_root
num_example_train_val = 6667
num_example_test = 3333
num_classes = 100
mean_per_class = True


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = FGVCAircraft(root, download=True, annotation_level='variant', split='trainval',transform=transform)
    return (dataset_train_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = FGVCAircraft(root, download=True, annotation_level='variant', split='test', transform=transform)
    return (dataset_test, None)

