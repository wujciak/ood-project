import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

from utils.metrics import build_labels_scores


def save_scores_to_csv(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    for dataset_name, methods in results.items():
        pd.DataFrame(methods).to_csv(
            os.path.join(output_dir, f"scores_{dataset_name}.csv"), index=False
        )


def plot_score_histograms(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    ood_keys = [k for k in results if k != "ID"]
    for method in results["ID"]:
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
    id_scores = results["ID"]
    ood_keys = [k for k in results if k != "ID"]
    for ood_key in ood_keys:
        plt.figure(figsize=(7, 6))
        for method in id_scores:
            y_true, y_scores = build_labels_scores(
                id_scores[method], results[ood_key][method]
            )
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, label=method, linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate", fontsize=13)
        plt.ylabel("True Positive Rate", fontsize=13)
        plt.title(f"ROC Curves: {ood_key.replace('_', ' ')}", fontsize=14)
        plt.legend(fontsize=11)
        plt.tick_params(labelsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"roc_{ood_key}.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()


def plot_pr_curves(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    id_scores = results["ID"]
    ood_keys = [k for k in results if k != "ID"]
    for ood_key in ood_keys:
        first = next(iter(id_scores))
        baseline = len(results[ood_key][first]) / (
            len(id_scores[first]) + len(results[ood_key][first])
        )
        plt.figure(figsize=(7, 6))
        for method in id_scores:
            y_true, y_scores = build_labels_scores(
                id_scores[method], results[ood_key][method]
            )
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            aupr = average_precision_score(y_true, y_scores)
            plt.plot(
                recall, precision, label=f"{method} (AUPR={aupr:.3f})", linewidth=2
            )
        plt.axhline(
            y=baseline,
            color="k",
            linestyle="--",
            linewidth=1,
            label=f"Random baseline ({baseline:.3f})",
        )
        plt.xlabel("Recall", fontsize=13)
        plt.ylabel("Precision", fontsize=13)
        plt.title(f"Precision-Recall Curves: {ood_key.replace('_', ' ')}", fontsize=14)
        plt.legend(fontsize=10)
        plt.tick_params(labelsize=11)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"pr_{ood_key}.png"), dpi=200, bbox_inches="tight"
        )
        plt.close()


def plot_auroc_bar(auroc_df, aupr_df, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    methods = auroc_df["method"].unique()
    ood_keys = auroc_df["dataset"].unique()
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (df, metric_name) in zip(axes, [(auroc_df, "AUROC"), (aupr_df, "AUPR")]):
        col = metric_name.lower()
        for i, ood_key in enumerate(ood_keys):
            vals = [
                df.loc[(df["method"] == m) & (df["dataset"] == ood_key), col].values[0]
                for m in methods
            ]
            offset = (i - (len(ood_keys) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=ood_key.replace("_", " "))
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(metric_name, fontsize=13)
        ax.set_title(f"{metric_name} by Method and OOD Scenario", fontsize=13)
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "auroc_aupr_bar.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()
