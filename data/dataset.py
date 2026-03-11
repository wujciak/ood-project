import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from config.config import CONFIG


def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    n_samples = 0
    mean = 0.0
    M2 = 0.0  # sum of squared deviations (for Welford-style variance)

    # First pass: compute global mean
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)  # (B, C, H*W)
        mean += images.mean(2).sum(0)
        n_samples += images.size(0)
    mean /= n_samples

    # Second pass: compute global variance
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)  # (B, C, H*W)
        M2 += ((images - mean.unsqueeze(0).unsqueeze(2)) ** 2).mean(2).sum(0)
    std = (M2 / n_samples).sqrt()

    return mean.tolist(), std.tolist()


def get_dataloaders():
    info_chest = INFO["chestmnist"]

    # Compute normalization stats from training set (ToTensor only, no augmentation)
    mean, std = compute_mean_std(
        medmnist.ChestMNIST(
            split="train", transform=transforms.ToTensor(), download=True
        )
    )

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = medmnist.ChestMNIST(
        split="train", transform=train_transform, download=True
    )
    test_dataset_id = medmnist.ChestMNIST(
        split="test", transform=data_transform, download=True
    )

    # Far-OOD: PathMNIST (histology, RGB → grayscale, normalized with ID stats)
    far_ood_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_dataset_ood_far = medmnist.PathMNIST(
        split="test", transform=far_ood_transform, download=True
    )

    # Near-OOD: PneumoniaMNIST (chest X-rays, different source hospital)
    # Already grayscale (1-channel), normalized with ChestMNIST stats
    test_dataset_ood_near = medmnist.PneumoniaMNIST(
        split="test", transform=data_transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    test_loader_id = DataLoader(
        test_dataset_id, batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader_ood_far = DataLoader(
        test_dataset_ood_far, batch_size=CONFIG["batch_size"], shuffle=False
    )
    test_loader_ood_near = DataLoader(
        test_dataset_ood_near, batch_size=CONFIG["batch_size"], shuffle=False
    )

    return (
        train_loader,
        test_loader_id,
        test_loader_ood_far,
        test_loader_ood_near,
        info_chest["n_channels"],
        len(info_chest["label"]),
    )
