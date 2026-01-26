import torch

CONFIG = {
    "batch_size": 64,
    "lr": 0.0001,
    "epochs": 30,  # do zmiany na więcej potem (np 30)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_ensemble": 5,
    "mc_samples": 15,
    "energy_temp": 1.0,
    "hybrid_weights": {"entropy": 0.7, "energy": 0.3},
}
