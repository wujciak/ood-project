import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def save_scores_to_csv(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, methods in results.items():
        df = pd.DataFrame(methods)
        df.to_csv(os.path.join(output_dir, f"scores_{dataset_name}.csv"), index=False)


def plot_score_histograms(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    ood_keys = [k for k in results if k != "ID"]

    for method in results["ID"].keys():
        plt.figure()
        plt.hist(results["ID"][method], bins=50, alpha=0.6, label="ID")
        for key in ood_keys:
            plt.hist(
                results[key][method], bins=50, alpha=0.6, label=key.replace("_", " ")
            )
        plt.legend()
        plt.title(f"Uncertainty Score Distribution: {method}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"hist_{method}.png"))
        plt.close()


def plot_roc_curves(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    id_scores = {method: results["ID"][method] for method in results["ID"]}
    ood_keys = [k for k in results if k != "ID"]

    for ood_key in ood_keys:
        plt.figure()
        for method in results["ID"].keys():
            y_true = np.concatenate(
                [
                    np.zeros(len(id_scores[method])),
                    np.ones(len(results[ood_key][method])),
                ]
            )
            y_scores = np.concatenate([id_scores[method], results[ood_key][method]])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, label=method)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves: {ood_key.replace('_', ' ')}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"roc_{ood_key}.png"))
        plt.close()
