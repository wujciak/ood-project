import torch
import torch.nn.functional as F


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
    """Negative free energy for multi-label (independent binary) classification.
    Based on per-label log-partition functions: score = T * sum_i softplus(f_i / T).
    Higher value = more extreme logits = more OOD-like.
    """
    return T * torch.sum(F.softplus(logits / T), dim=1)


def compute_mutual_information(all_probs):
    """BALD (Bayesian Active Learning by Disagreement) score.
    Mutual information between predictions and model parameters:
    MI = H[E_k[p_k]] - E_k[H[p_k]]  (predictive entropy minus expected entropy).

    Args:
        all_probs: (K, batch_size, num_classes) sigmoid probabilities from
                    K stochastic forward passes (MC Dropout or Ensemble members).
    Returns:
        (batch_size,) mutual information scores.
        Higher = more model disagreement = more OOD-like.
    """
    epsilon = 1e-10

    # Predictive entropy: H[E_k[p_k]]
    mean_probs = torch.mean(all_probs, dim=0)
    pred_entropy = -(
        mean_probs * torch.log(mean_probs + epsilon)
        + (1 - mean_probs) * torch.log(1 - mean_probs + epsilon)
    )
    pred_entropy = torch.mean(pred_entropy, dim=1)

    # Expected entropy: E_k[H[p_k]]
    per_sample_entropy = -(
        all_probs * torch.log(all_probs + epsilon)
        + (1 - all_probs) * torch.log(1 - all_probs + epsilon)
    )
    expected_entropy = torch.mean(torch.mean(per_sample_entropy, dim=2), dim=0)

    return pred_entropy - expected_entropy
