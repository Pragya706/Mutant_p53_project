"""
src/evaluation/metrics.py
──────────────────────────
Evaluation + visualisation for 5-class p53 pathogenicity prediction.
Class labels: Benign / VUS / Possibly pathogenic / Likely Pathogenic / Pathogenic
"""

import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             classification_report)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import OUTPUT_DIR as RESULTS_DIR, LABEL_ORDER

log = logging.getLogger(__name__)
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]


def compute_metrics(y_true, y_pred, y_proba,
                    model_name: str = "Model") -> dict:
    # Handle binary (n,2) or multi-class proba
    if hasattr(y_proba, "ndim") and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    metrics = {
        "model":     model_name,
        "auc":       round(auc, 4),
        "f1":        round(f1_score(y_true , y_pred , average="weighted"), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }
    log.info(f"{model_name}: AUC={metrics['auc']}  "
             f"F1={metrics['f1']}  Acc={metrics['accuracy']}")
    return metrics


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("model")
    path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(path)
    log.info(f"Model comparison saved -> {path}")
    log.info("\n" + df.to_string())
    return df


def plot_confusion_matrix(y_true, y_pred,
                          model_name: str = "Ensemble",
                          save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"cm_{model_name.lower()}.png")
        fig.savefig(path, dpi=150)
        log.info(f"Confusion matrix saved -> {path}")
    plt.close()


def plot_class_roc(y_true, y_proba, model_name="Ensemble", save=True):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, roc_auc_score

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # FIX
    if len(y_proba.shape) == 1:
        y_proba = y_proba.reshape(-1, 1)

    if y_proba.shape[1] == 1:
        y_proba = np.hstack([1 - y_proba, y_proba])

    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(9, 6))

    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC - {model_name}")
    ax.legend()

    if save:
        path = os.path.join("results", f"roc_{model_name.lower()}.png")
        plt.savefig(path)

    plt.close()


def plot_feature_importance(importance_dict: dict, top_n: int = 25,
                            title: str = "Feature Importance",
                            save: bool = True):
    items  = sorted(importance_dict.items(),
                    key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(list(names)[::-1], list(values)[::-1], color=PALETTE[0])
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "feature_importance.png")
        fig.savefig(path, dpi=150)
        log.info(f"Feature importance saved -> {path}")
    plt.close()


def plot_target_ranking(ranking_df: pd.DataFrame,
                        codon: int, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE[3] if i == 0 else PALETTE[0]
              for i in range(len(ranking_df))]
    ax.barh(ranking_df["protein"][::-1],
            ranking_df["final_score"][::-1],
            color=colors[::-1])
    ax.set_xlabel("Restoration Score")
    ax.set_title(f"Protein Target Ranking — Codon {codon} Mutation")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"target_ranking_{codon}.png")
        fig.savefig(path, dpi=150)
        log.info(f"Target ranking saved -> {path}")
    plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, save: bool = True):
    metrics = ["auc", "f1", "accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        comparison_df[metric].plot(kind="bar", ax=ax,
                                   color=PALETTE[:len(comparison_df)],
                                   edgecolor="white", width=0.6)
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Model Performance Comparison – p53 Pathogenicity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150)
        log.info(f"Model comparison saved -> {path}")
    plt.close()


def plot_pathogenicity_distribution(y_true, y_pred,
                                    save: bool = True):
    """Bar chart showing true vs predicted class distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arr, title in [
        (axes[0], y_true, "True Distribution"),
        (axes[1], y_pred, "Predicted Distribution"),
    ]:
        counts = [np.sum(arr == i) for i in range(len(LABEL_ORDER))]
        ax.bar(LABEL_ORDER, counts,
               color=PALETTE[:len(LABEL_ORDER)], edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Pathogenicity Class Distribution", fontsize=13)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "class_distribution.png")
        fig.savefig(path, dpi=150)
        log.info(f"Class distribution saved -> {path}")
    plt.close()
