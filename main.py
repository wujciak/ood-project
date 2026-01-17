import numpy as np
from sklearn.metrics import roc_auc_score

from config import CONFIG
from models import ResNet18_MCDropout
from data import get_dataloaders
from utils import train_model, get_uncertainty_deterministic, get_uncertainty_mc_dropout, get_uncertainty_ensemble


def main():
    print(f"Device: {CONFIG['device']}")
    
    train_loader, test_id, test_ood_cross, test_ood_synth, n_channels, n_classes = get_dataloaders()
    
    ensemble_models = []
    print("\n--- Training Deep Ensembles ---")
    for i in range(CONFIG['n_ensemble']):
        print(f"Training Model {i+1}/{CONFIG['n_ensemble']}")
        model = ResNet18_MCDropout(in_channels=n_channels, num_classes=n_classes).to(CONFIG['device'])
        
        # 1 epoch for demo speed we will increase this later
        for epoch in range(1): 
            train_model(model, train_loader, CONFIG['device'])
        ensemble_models.append(model)

    single_model = ensemble_models[0]

    print("\n--- Computing Uncertainty Scores ---")
    datasets = {'ID': test_id, 'OOD_Cross': test_ood_cross, 'OOD_Synth': test_ood_synth}
    results = {}

    for name, loader in datasets.items():
        print(f"Processing {name}...")
        res = {}
        
        det_scores = get_uncertainty_deterministic(single_model, loader, CONFIG['device'])
        res['entropy'] = np.array(det_scores['entropy'])
        res['energy'] = np.array(det_scores['energy'])
        res['hybrid'] = (CONFIG['hybrid_weights']['entropy'] * res['entropy'] + 
                         CONFIG['hybrid_weights']['energy'] * res['energy'])
        res['mc_dropout'] = get_uncertainty_mc_dropout(single_model, loader, CONFIG['device'])
        res['ensemble'] = get_uncertainty_ensemble(ensemble_models, loader, CONFIG['device'])
        
        results[name] = res

    print("\n--- OOD Detection Performance (AUROC) ---")
    # OOD is positive class (1), ID is negative (0)
    methods = ['entropy', 'energy', 'hybrid', 'mc_dropout', 'ensemble']
    
    print(f"{'Method':<15} | {'Cross-Dataset OOD':<20} | {'Synthetic OOD':<20}")
    print("-" * 60)
    
    for method in methods:
        id_scores = results['ID'][method]
        
        # Cross Dataset
        ood_cross = results['OOD_Cross'][method]
        y_true_cross = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_cross))])
        y_scores_cross = np.concatenate([id_scores, ood_cross])
        auroc_cross = roc_auc_score(y_true_cross, y_scores_cross)
        
        # Synthetic
        ood_synth = results['OOD_Synth'][method]
        y_true_synth = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_synth))])
        y_scores_synth = np.concatenate([id_scores, ood_synth])
        auroc_synth = roc_auc_score(y_true_synth, y_scores_synth)
        
        print(f"{method:<15} | {auroc_cross:.4f}{' ':14} | {auroc_synth:.4f}")

if __name__ == "__main__":
    main()
