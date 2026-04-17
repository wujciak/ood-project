import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from datetime import datetime

from config import CONFIG
from models import ResNet18_MCDropout
from data import get_dataloaders
from utils import (
    train_model,
    get_uncertainty_deterministic,
    get_uncertainty_mc_dropout,
    get_uncertainty_ensemble,
)
from utils.metrics import build_labels_scores
from utils.save_results import (
    save_scores_to_csv,
    plot_score_histograms,
    plot_roc_curves,
    plot_pr_curves,
    plot_auroc_bar,
)

METHODS = ["entropy", "energy", "hybrid", "mc_dropout", "ensemble"]


def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/csv", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = f"results/csv/{timestamp}"
    plots_dir = f"results/plots/{timestamp}"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Device: {CONFIG['device']}")

    train_loader, test_id, test_ood_far, test_ood_near, n_channels, n_classes = (
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
            model.load_state_dict(
                torch.load(model_path, map_location=CONFIG["device"], weights_only=True)
            )
            print(f"Loaded saved model {i+1} from {model_path}")
        else:
            print(f"Training Model {i+1}/{CONFIG['n_ensemble']}")
            train_model(model, train_loader, CONFIG["device"])
            torch.save(model.state_dict(), model_path)
            print(f"  Saved model to {model_path}")

        ensemble_models.append(model)

    single_model = ensemble_models[0]

    print("\n--- Computing Uncertainty Scores ---")
    datasets = {"ID": test_id, "OOD_Far": test_ood_far, "OOD_Near": test_ood_near}
    results = {}

    for name, loader in datasets.items():
        print(f"Processing {name}...")
        res = {}

        det_scores = get_uncertainty_deterministic(
            single_model, loader, CONFIG["device"]
        )
        res["entropy"] = np.array(det_scores["entropy"])
        res["energy"] = np.array(det_scores["energy"])
        res["mc_dropout"] = get_uncertainty_mc_dropout(
            single_model, loader, CONFIG["device"]
        )
        res["ensemble"] = get_uncertainty_ensemble(
            ensemble_models, loader, CONFIG["device"]
        )

        results[name] = res

    # Compute hybrid score with global min-max normalization across all datasets,
    # so that ID and OOD scores share the same scale before combining (as per paper).
    all_entropy = np.concatenate([results[n]["entropy"] for n in datasets])
    all_energy = np.concatenate([results[n]["energy"] for n in datasets])
    ent_min, ent_max = all_entropy.min(), all_entropy.max()
    eng_min, eng_max = all_energy.min(), all_energy.max()

    for name in datasets:
        norm_ent = (results[name]["entropy"] - ent_min) / (ent_max - ent_min + 1e-10)
        norm_eng = (results[name]["energy"] - eng_min) / (eng_max - eng_min + 1e-10)
        results[name]["hybrid"] = (
            CONFIG["hybrid_weights"]["entropy"] * norm_ent
            + CONFIG["hybrid_weights"]["energy"] * norm_eng
        )

    # Save results CSVs
    save_scores_to_csv(results, output_dir=csv_dir)

    # Save mean/std summary
    summary = []
    for method in METHODS:
        for dataset in ["ID", "OOD_Far", "OOD_Near"]:
            scores = results[dataset][method]
            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "mean": scores.mean(),
                    "std": scores.std(),
                }
            )
    pd.DataFrame(summary).to_csv(f"{csv_dir}/score_statistics.csv", index=False)

    # Compute and save AUROC & AUPR
    # All scores: higher = more uncertain = more OOD-like (label 1), ID = label 0
    auroc_summary, aupr_summary = [], []
    for method in METHODS:
        id_scores = results["ID"][method]
        for dataset_name, ood_scores in [
            ("OOD_Far", results["OOD_Far"][method]),
            ("OOD_Near", results["OOD_Near"][method]),
        ]:
            y_true, y_scores = build_labels_scores(id_scores, ood_scores)
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

    auroc_df = pd.DataFrame(auroc_summary)
    aupr_df = pd.DataFrame(aupr_summary)
    auroc_df.to_csv(f"{csv_dir}/auroc.csv", index=False)
    aupr_df.to_csv(f"{csv_dir}/aupr.csv", index=False)

    # Plot histograms, ROC curves, PR curves and bar chart
    plot_score_histograms(results, output_dir=plots_dir)
    plot_roc_curves(results, output_dir=plots_dir)
    plot_pr_curves(results, output_dir=plots_dir)
    plot_auroc_bar(auroc_df, aupr_df, output_dir=plots_dir)

    print("\n--- All results saved ---")
    print("Models: results/models/")
    print(f"CSV tables: {csv_dir}/")
    print(f"Plots: {plots_dir}/")


if __name__ == "__main__":
    main()
