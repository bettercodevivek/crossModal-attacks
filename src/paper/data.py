"""MNIST and CIFAR-10 dataloaders with train/test splits."""
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .paper_config import PROJECT_ROOT


def get_loaders(
    dataset: str,
    batch_size: int,
    num_workers: int = 0,
    data_root: str = None,
    train_max_samples: int = None,
):
    """
    train_max_samples: if set, only the first N training images are used (faster benchmarks).
    """
    data_root = data_root or os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_root, exist_ok=True)
    dataset = dataset.lower()
    if dataset == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(
            root=data_root, train=True, download=True, transform=tfm
        )
        test_ds = datasets.MNIST(
            root=data_root, train=False, download=True, transform=tfm
        )
    elif dataset == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=tfm
        )
        test_ds = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=tfm
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if train_max_samples is not None:
        n = min(train_max_samples, len(train_ds))
        train_ds = Subset(train_ds, range(n))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
