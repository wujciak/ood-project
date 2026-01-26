import os
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

    for method in results["ID"].keys():
        plt.figure()
        plt.hist(results["ID"][method], bins=50, alpha=0.6, label="ID")
        plt.hist(results["OOD_Cross"][method], bins=50, alpha=0.6, label="OOD Cross")
        plt.hist(results["OOD_Synth"][method], bins=50, alpha=0.6, label="OOD Synth")
        plt.legend()
        plt.title(f"Uncertainty Score Distribution: {method}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"hist_{method}.png"))
        plt.close()


def plot_roc_curves(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    for method in results["ID"].keys():
        # Cross-dataset
        id_scores = results["ID"][method]
        ood_scores = results["OOD_Cross"][method]

        y_true = [0] * len(id_scores) + [1] * len(ood_scores)
        y_scores = list(id_scores) + list(ood_scores)

        fpr, tpr, _ = roc_curve(y_true, y_scores)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{method} (Cross OOD)")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {method}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"roc_{method}.png"))
        plt.close()
