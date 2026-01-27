import torch


def compute_predictive_entropy(probs):
    """Mean binary entropy across labels for multi-label task.
    Higher entropy = more uncertain.
    """
    epsilon = 1e-10
    entropy = -(probs * torch.log(probs + epsilon) + (1 - probs) * torch.log(
        1 - probs + epsilon
    ))
    return torch.mean(entropy, dim=1)


def compute_energy_score(logits, T=1.0):
    """Energy score for multi-label classification using LogSumExp.
    Lower (more negative) energy = more uncertain/OOD.
    """
    return -T * torch.logsumexp(logits / T, dim=1)
