import torch

CONFIG = {
    "batch_size": 64,
    "lr": 0.0001,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_ensemble": 5,
    "mc_samples": 30,
    "energy_temp": 0.5,
    "hybrid_weights": {"entropy": 0.5, "energy": 0.5},
}
