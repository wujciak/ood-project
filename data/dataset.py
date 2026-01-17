import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from config.config import CONFIG


class SyntheticNoiseDataset(Dataset):
    def __init__(self, base_dataset, noise_factor=0.2):
        self.base = base_dataset
        self.noise_factor = noise_factor
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, target = self.base[idx]
        noise = torch.randn_like(img) * self.noise_factor
        img_noisy = torch.clamp(img + noise, 0, 1)
        return img_noisy, target


def get_dataloaders():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # In-distribution: ChestMNIST
    info_chest = INFO['chestmnist']
    train_dataset = medmnist.ChestMNIST(split='train', transform=data_transform, download=True)
    test_dataset_id = medmnist.ChestMNIST(split='test', transform=data_transform, download=True)
    
    # Cross-dataset OOD: PathMNIST (convert RGB to grayscale)
    ood_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_dataset_ood_cross = medmnist.PathMNIST(split='test', transform=ood_transform, download=True)

    # Synthetic OOD: noise injection
    test_dataset_ood_synth = SyntheticNoiseDataset(test_dataset_id)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader_id = DataLoader(test_dataset_id, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader_ood_cross = DataLoader(test_dataset_ood_cross, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader_ood_synth = DataLoader(test_dataset_ood_synth, batch_size=CONFIG['batch_size'], shuffle=False)

    return train_loader, test_loader_id, test_loader_ood_cross, test_loader_ood_synth, info_chest['n_channels'], len(info_chest['label'])
