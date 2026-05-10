"""
MNIST data loaders. The notebook subsamples 10k train and 2k test for
tractability of the inner Langevin loops.
"""

import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)


def get_mnist_loaders(data_dir='./data', n_train=10_000, n_test=2_000,
                      batch_size=128, seed=42):
    """Returns (train_loader, test_loader, input_dim).

    The subset indices are drawn with a numpy RNG seeded by `seed` so the
    split is reproducible across runs.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten),
    ])

    train_full = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform,
    )
    test_full = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform,
    )

    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(train_full), n_train, replace=False)
    test_idx = rng.choice(len(test_full), n_test, replace=False)

    train_loader = DataLoader(
        Subset(train_full, train_idx), batch_size=batch_size,
        shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        Subset(test_full, test_idx), batch_size=batch_size, shuffle=False,
    )
    log.info(f'MNIST: train={len(train_idx)}  test={len(test_idx)}  input_dim=784')
    return train_loader, test_loader, 784
