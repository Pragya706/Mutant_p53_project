"""
train_gnn.py  –  Training loop for GATClassifier on TP53 mutation graph

Improvements vs baseline
─────────────────────────
  • Focal loss instead of cross-entropy -> better handles class imbalance
  • Cosine-annealing LR schedule with warm restarts
  • Early stopping on validation AUC
  • Saves best checkpoint (best_gnn.pt)
  • Full classification report + ROC-AUC at the end
"""

import os
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GNN, OUTPUT_DIR, MODEL_DIR, DEVICE
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
from src.graph_model.gnn_model import GATClassifier, FocalLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(MODEL_DIR, "best_gnn.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Train / eval helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, data, optimiser, criterion, device):
    model.train()
    optimiser.zero_grad()
    logits = model(data.x.to(device), data.edge_index.to(device))
    mask   = data.train_mask.to(device)
    loss   = criterion(logits[mask], data.y.to(device)[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimiser.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask_name, device):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    mask   = getattr(data, mask_name).to(device)
    labels = data.y.to(device)[mask].cpu().numpy()
    probs  = torch.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    preds  = (probs >= 0.5).astype(int)

    auc       = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1        = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall,
            "probs": probs, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(data):
    """
    Full training loop.
    Parameters
    ----------
    data : torch_geometric.data.Data   (from gnn_dataset.build_graph_data)

    Returns
    -------
    model     : trained GATClassifier
    test_metrics : dict with AUC, F1, precision, recall
    """
    device = torch.device(DEVICE)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GATClassifier(
        in_features = data.x.shape[1],
        hidden_dim  = GNN["hidden_dim"],
        num_heads   = GNN["num_heads"],
        num_layers  = GNN["num_layers"],
        dropout     = GNN["dropout"],
        num_classes = 2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GATClassifier trainable parameters: {n_params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    criterion = FocalLoss(gamma=2.0, alpha=0.75)
    optimiser = optim.AdamW(model.parameters(),
                             lr=GNN["lr"],
                             weight_decay=GNN["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=30, T_mult=2, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc  = 0.0
    patience_left = GNN["patience"]
    history       = {"train_loss": [], "val_auc": [], "val_f1": []}

    logger.info("=== GNN Training ===")
    for epoch in range(1, GNN["epochs"] + 1):
        loss = train_epoch(model, data, optimiser, criterion, device)
        scheduler.step(epoch)

        if epoch % 5 == 0 or epoch == 1:
            val_m = evaluate(model, data, "val_mask", device)
            history["train_loss"].append(loss)
            history["val_auc"].append(val_m["auc"])
            history["val_f1"].append(val_m["f1"])

            logger.info(
                f"  Epoch {epoch:3d}/{GNN['epochs']} | "
                f"Loss={loss:.4f} | "
                f"Val AUC={val_m['auc']:.4f} | "
                f"Val F1={val_m['f1']:.4f} | "
                f"Val Prec={val_m['precision']:.4f} | "
                f"Val Recall={val_m['recall']:.4f}"
            )

            if val_m["auc"] > best_val_auc:
                best_val_auc  = val_m["auc"]
                patience_left = GNN["patience"]
                torch.save(model.state_dict(), CKPT_PATH)
            else:
                patience_left -= 1
                if patience_left == 0:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

    # ── Load best checkpoint ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    logger.info(f"Best checkpoint loaded (val AUC={best_val_auc:.4f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_m = evaluate(model, data, "test_mask", device)
    logger.info("\n=== GNN TEST RESULTS ===")
    logger.info(f"  AUC       : {test_m['auc']:.4f}")
    logger.info(f"  F1-score  : {test_m['f1']:.4f}")
    logger.info(f"  Precision : {test_m['precision']:.4f}")
    logger.info(f"  Recall    : {test_m['recall']:.4f}")

    preds = (test_m["probs"] >= 0.5).astype(int)
    print("\nClassification Report (GNN – Test Set):")
    print(classification_report(test_m["labels"], preds,
                                 target_names=["Benign/VUS", "Pathogenic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(test_m["labels"], preds))

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_training_curves(history)
    _plot_roc(test_m["labels"], test_m["probs"])

    return model, test_m


def _plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = list(range(5, 5 * len(history["train_loss"]) + 5, 5))

    axes[0].plot(epochs, history["train_loss"], color="#e74c3c", linewidth=2)
    axes[0].set_title("Training Loss (Focal)", fontsize=13)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_auc"],  label="AUC",  color="#2ecc71", linewidth=2)
    axes[1].plot(epochs, history["val_f1"],   label="F1",   color="#3498db", linewidth=2)
    axes[1].set_title("Validation Metrics", fontsize=13)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gnn_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Training curves saved -> {path}")


def _plot_roc(labels, probs):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#e74c3c", linewidth=2.5,
             label=f"GAT Classifier  (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – GNN Test Set")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "gnn_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ROC curve saved -> {path}")



# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper called by run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def predict_gnn(model, data):
    """Return softmax probabilities for all test-mask nodes. Shape: (n_test, 2)."""
    import torch
    device = torch.device(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    # Return only test-mask rows
    mask = data.test_mask.cpu().numpy()
    return probs[mask]   # shape (n_test, 2)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from feature_engineering import run_pipeline
    from gnn_dataset import build_graph_data

    res  = run_pipeline()
    data = build_graph_data(res["X_scaled"], res["y"], res["df_raw"])
    model, metrics = train_gnn(data)
