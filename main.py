import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch

from config import CONFIG
from models import ResNet18_MCDropout
from data import get_dataloaders
from utils import (
    train_model,
    get_uncertainty_deterministic,
    get_uncertainty_mc_dropout,
    get_uncertainty_ensemble,
)
from utils.save_results import (
    save_scores_to_csv,
    plot_score_histograms,
    plot_roc_curves,
)

# Ensure results folders exist
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/csv", exist_ok=True)


def main():
    print(f"Device: {CONFIG['device']}")

    train_loader, test_id, test_ood_cross, test_ood_synth, n_channels, n_classes = (
        get_dataloaders()
    )

    ensemble_models = []
    print("\n--- Preparing Deep Ensembles ---")
    for i in range(CONFIG["n_ensemble"]):
        model_path = f"results/models/model_{i+1}.pt"
        model = ResNet18_MCDropout(in_channels=n_channels, num_classes=n_classes).to(
            CONFIG["device"]
        )

        if os.path.exists(model_path):
            # Load previously trained model
            model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"]))
            print(f"Loaded saved model {i+1} from {model_path}")
        else:
            # Train model if not saved
            print(f"Training Model {i+1}/{CONFIG['n_ensemble']}")
            for epoch in range(CONFIG["epochs"]):
                print(f"  Epoch {epoch+1}/{CONFIG['epochs']}")
                loss = train_model(model, train_loader, CONFIG["device"])
                print(f"    Loss: {loss:.4f}")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path}")

        ensemble_models.append(model)

    single_model = ensemble_models[0]

    print("\n--- Computing Uncertainty Scores ---")
    datasets = {"ID": test_id, "OOD_Cross": test_ood_cross, "OOD_Synth": test_ood_synth}
    results = {}

    for name, loader in datasets.items():
        print(f"Processing {name}...")
        res = {}

        det_scores = get_uncertainty_deterministic(
            single_model, loader, CONFIG["device"]
        )
        res["entropy"] = np.array(det_scores["entropy"])
        res["energy"] = np.array(det_scores["energy"])
        res["hybrid"] = (
            CONFIG["hybrid_weights"]["entropy"] * res["entropy"]
            + CONFIG["hybrid_weights"]["energy"] * res["energy"]
        )
        res["mc_dropout"] = get_uncertainty_mc_dropout(
            single_model, loader, CONFIG["device"]
        )
        res["ensemble"] = get_uncertainty_ensemble(
            ensemble_models, loader, CONFIG["device"]
        )

        results[name] = res

    # Save results CSVs
    save_scores_to_csv(results, output_dir="results/csv")

    # Save mean/std summary
    summary = []
    for method in ["entropy", "energy", "hybrid", "mc_dropout", "ensemble"]:
        for dataset in ["ID", "OOD_Cross", "OOD_Synth"]:
            scores = results[dataset][method]
            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "mean": scores.mean(),
                    "std": scores.std(),
                }
            )
    pd.DataFrame(summary).to_csv("results/csv/score_statistics.csv", index=False)

    # Compute and save AUROC & AUPR
    auroc_summary, aupr_summary = [], []
    for method in ["entropy", "energy", "hybrid", "mc_dropout", "ensemble"]:
        id_scores = results["ID"][method]
        for dataset_name, ood_scores in [
            ("OOD_Cross", results["OOD_Cross"][method]),
            ("OOD_Synth", results["OOD_Synth"][method]),
        ]:
            y_true = np.concatenate(
                [np.zeros(len(id_scores)), np.ones(len(ood_scores))]
            )
            y_scores = np.concatenate([id_scores, ood_scores])
            auroc_summary.append(
                {
                    "method": method,
                    "dataset": dataset_name,
                    "auroc": roc_auc_score(y_true, y_scores),
                }
            )
            aupr_summary.append(
                {
                    "method": method,
                    "dataset": dataset_name,
                    "aupr": average_precision_score(y_true, y_scores),
                }
            )

    pd.DataFrame(auroc_summary).to_csv("results/csv/auroc.csv", index=False)
    pd.DataFrame(aupr_summary).to_csv("results/csv/aupr.csv", index=False)

    # Plot histograms and ROC curves
    plot_score_histograms(results, output_dir="results/plots")
    plot_roc_curves(results, output_dir="results/plots")

    print("\n--- All results saved ---")
    print("Models: results/models/")
    print("CSV tables: results/csv/")
    print("Plots: results/plots/")


if __name__ == "__main__":
    main()
