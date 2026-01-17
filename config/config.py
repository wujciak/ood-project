import torch


CONFIG = {
    'batch_size': 64,
    'lr': 0.0001,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_ensemble': 3,
    'mc_samples': 5,
    'energy_temp': 1.0,
    'hybrid_weights': {'entropy': 0.7, 'energy': 0.3}
}
