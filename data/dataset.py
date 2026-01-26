import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from config.config import CONFIG


class SyntheticOODDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.blur = transforms.GaussianBlur(kernel_size=5)
        self.rotate = transforms.RandomRotation(degrees=30)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]

        # Randomly choose perturbation
        r = np.random.rand()
        if r < 0.33:
            noise = torch.randn_like(img) * 0.5
            img = torch.clamp(img + noise, -2, 2)
        elif r < 0.66:
            img = self.rotate(img)
        else:
            img = self.blur(img)

        return img, target


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += images.size(0)

    mean /= n_samples
    std /= n_samples
    return mean.tolist(), std.tolist()


def get_dataloaders():
    # Load raw dataset to compute normalization
    base_transform = transforms.ToTensor()
    train_dataset_raw = medmnist.ChestMNIST(
        split="train", transform=base_transform, download=True
    )

    mean, std = compute_mean_std(train_dataset_raw)

    # Training transform with augmentation
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    
    # Test transform without augmentation
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    # In-distribution: ChestMNIST
    info_chest = INFO["chestmnist"]
    train_dataset = medmnist.ChestMNIST(
        split="train", transform=train_transform, download=True
    )
    test_dataset_id = medmnist.ChestMNIST(
        split="test", transform=data_transform, download=True
    )

    # Cross-dataset OOD: PathMNIST (convert RGB to grayscale)
    ood_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_dataset_ood_cross = medmnist.PathMNIST(
        split="test", transform=ood_transform, download=True
    )

    # Synthetic OOD
    test_dataset_ood_synth = SyntheticOODDataset(test_dataset_id)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    test_loader_id = DataLoader(
        test_dataset_id, batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader_ood_cross = DataLoader(
        test_dataset_ood_cross, batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader_ood_synth = DataLoader(
        test_dataset_ood_synth, batch_size=CONFIG["batch_size"], shuffle=False
    )

    return (
        train_loader,
        test_loader_id,
        test_loader_ood_cross,
        test_loader_ood_synth,
        info_chest["n_channels"],
        len(info_chest["label"]),
    )
