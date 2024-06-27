from typing import Tuple
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import Flowers102
import torchvision
from .utils import dataset_root

root = dataset_root
num_example_train_val = 2040
num_example_test = 6149
num_classes = 102
mean_per_class = True


class Warper(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if torchvision.__version__ >= "0.13.0":
            return img, label
        else:
            return img, label - 1


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = Flowers102(root, download=True, split='train', transform=transform)
    dataset_val = Flowers102(root, download=True, split='val', transform=transform)
    dataset_train = ConcatDataset([dataset_train, dataset_val])
    dataset_train = Warper(dataset_train)
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = Flowers102(root, download=True, split='test', transform=transform)
    dataset_test = Warper(dataset_test)
    return (dataset_test, None)

