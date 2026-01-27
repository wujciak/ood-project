import torch


def compute_predictive_entropy(probs):
    """Mean binary entropy across labels for multi-label task.
    Higher entropy = more uncertain.
    """
    epsilon = 1e-10
    entropy = -(
        probs * torch.log(probs + epsilon)
        + (1 - probs) * torch.log(1 - probs + epsilon)
    )
    return torch.mean(entropy, dim=1)


def compute_energy_score(logits, T=1.0):
    """Energy-based uncertainty for multi-label classification.
    Uses max absolute logit magnitude as uncertainty measure.
    Higher score = more extreme predictions = more OOD-like.
    """
    # Large magnitude logits (extreme predictions) indicate OOD
    # Small magnitude logits (moderate predictions) indicate ID
    return torch.max(torch.abs(logits), dim=1)[0]
