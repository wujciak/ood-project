import torch
import numpy as np

from config.config import CONFIG
from utils.metrics import (
    compute_predictive_entropy,
    compute_energy_score,
    compute_mutual_information,
)


def get_uncertainty_deterministic(model, loader, device):
    """Standard Forward Pass + Energy & Entropy"""
    model.eval()
    uncertainties = {"entropy": [], "energy": []}

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            ent = compute_predictive_entropy(probs)
            uncertainties["entropy"].extend(ent.cpu().numpy())

            eng = compute_energy_score(logits, CONFIG["energy_temp"])
            uncertainties["energy"].extend(eng.cpu().numpy())

    return uncertainties


def get_uncertainty_mc_dropout(model, loader, device, k=CONFIG["mc_samples"]):
    """Monte Carlo Dropout — returns mutual information (BALD) scores.
    MI captures model disagreement across stochastic forward passes.
    """
    model.train()  # Keep dropout active
    uncertainties = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            mc_probs = []
            for _ in range(k):
                logits = model(inputs)
                mc_probs.append(torch.sigmoid(logits))

            mc_probs = torch.stack(mc_probs)  # (K, batch, classes)
            mi = compute_mutual_information(mc_probs)
            uncertainties.extend(mi.cpu().numpy())

    return np.array(uncertainties)


def get_uncertainty_ensemble(ensemble_models, loader, device):
    """Deep Ensembles — returns mutual information scores.
    MI captures disagreement between independently trained models.
    """
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

            ensemble_probs = torch.stack(ensemble_probs)  # (K, batch, classes)
            mi = compute_mutual_information(ensemble_probs)
            uncertainties.extend(mi.cpu().numpy())

    return np.array(uncertainties)
