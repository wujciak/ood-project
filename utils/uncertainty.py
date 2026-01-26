import torch
import numpy as np

from config.config import CONFIG
from utils.metrics import compute_predictive_entropy, compute_energy_score


def get_uncertainty_deterministic(model, loader, device):
    """Standard Forward Pass + Energy & Entropy"""
    model.eval()
    uncertainties = {"entropy": [], "energy": [], "confidence": []}

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            ent = compute_predictive_entropy(probs)
            uncertainties["entropy"].extend(ent.cpu().numpy())

            eng = compute_energy_score(logits, CONFIG["energy_temp"])
            uncertainties["energy"].extend(eng.cpu().numpy())

            conf = torch.mean(torch.max(probs, 1 - probs), dim=1)
            uncertainties["confidence"].extend(conf.cpu().numpy())

    return uncertainties


def get_uncertainty_mc_dropout(model, loader, device, k=CONFIG["mc_samples"]):
    """Monte Carlo Dropout"""
    model.train()  # Keep dropout active
    uncertainties = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            mc_probs = []
            for _ in range(k):
                logits = model(inputs)
                mc_probs.append(torch.sigmoid(logits))

            mc_probs = torch.stack(mc_probs)
            mean_probs = torch.mean(mc_probs, dim=0)

            ent = compute_predictive_entropy(mean_probs)
            uncertainties.extend(ent.cpu().numpy())

    return np.array(uncertainties)


def get_uncertainty_ensemble(ensemble_models, loader, device):
    """Deep Ensembles"""
    for m in ensemble_models:
        m.eval()
    uncertainties = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            ensemble_probs = []
            for model in ensemble_models:
                logits = model(inputs)
                ensemble_probs.append(torch.sigmoid(logits))

            avg_probs = torch.mean(torch.stack(ensemble_probs), dim=0)

            ent = compute_predictive_entropy(avg_probs)
            uncertainties.extend(ent.cpu().numpy())

    return np.array(uncertainties)
