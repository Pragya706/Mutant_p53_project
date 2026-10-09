"""
fl_server.py  –  Federated Learning server (FedAvg aggregation)

Key improvements
─────────────────
  • Data partitioning by cancer type (Disease column) -> realistic heterogeneity
  • Weighted FedAvg aggregation (by client dataset size)
  • Per-round global evaluation
  • Convergence plot
  • Saves best global model
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FL, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED, DEVICE
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
from src.federated.fl_client import FLClient, LocalMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FL_CKPT = os.path.join(MODEL_DIR, "best_fl_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  Data partitioning
# ─────────────────────────────────────────────────────────────────────────────

def partition_by_cancer_type(
    X: np.ndarray, y: np.ndarray, df_raw,
    n_clients: int = None,
    min_samples: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Partition data into federated clients by cancer type (Disease column).
    Each unique cancer type (with >= min_samples) becomes one client.
    If fewer unique types are available than n_clients, fall back to random split.

    Returns list of (X_client, y_client) tuples.
    """
    if n_clients is None:
        n_clients = FL["num_clients"]

    partitions = []

    if df_raw is not None and "Disease" in df_raw.columns:
        disease = df_raw["Disease"].fillna("Unknown").str.strip()
        disease_aligned = disease.iloc[:len(y)]   # align index

        # Get diseases with enough samples
        counts  = disease_aligned.value_counts()
        eligible = counts[counts >= min_samples].index.tolist()
        logger.info(f"Eligible cancer types (>={min_samples} samples): {len(eligible)}")

        selected = eligible[:n_clients]
        for cancer in selected:
            mask = disease_aligned.values == cancer
            Xi, yi = X[mask], y[mask]
            if len(yi) >= min_samples:
                partitions.append((Xi, yi))
                logger.info(f"  Client '{cancer}': {len(yi)} samples, "
                            f"pos_rate={yi.mean():.3f}")

    # Fallback: random partition
    if len(partitions) < 2:
        logger.warning("Falling back to random data partitioning …")
        partitions = random_partition(X, y, n_clients)

    return partitions[:n_clients]


def random_partition(
    X: np.ndarray, y: np.ndarray, n_clients: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Randomly split data into n_clients roughly equal partitions."""
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(y))
    splits = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in splits]


# ─────────────────────────────────────────────────────────────────────────────
#  FedAvg aggregation
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(
    local_states: list[dict],
    weights: list[int],
) -> dict:
    """
    Weighted FedAvg: new_global_w = Σ (n_k / N) * w_k
    """
    total = sum(weights)
    global_state = copy.deepcopy(local_states[0])

    for key in global_state:
        global_state[key] = sum(
            (w / total) * state[key].float()
            for state, w in zip(local_states, weights)
        )

    return global_state


# ─────────────────────────────────────────────────────────────────────────────
#  Global evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def global_evaluate(
    global_state: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    in_features: int,
    device: str = "cpu",
) -> dict:
    device = torch.device(device)
    model = LocalMLP(in_features, FL["hidden_dims"], FL["dropout"]).to(device)
    model.load_state_dict(global_state)
    model.eval()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    logits = model(X_t)
    probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds  = (probs >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    f1        = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)

    return {"auc": auc, "f1": f1, "precision": precision, "recall": recall,
            "probs": probs, "preds": preds}


# ─────────────────────────────────────────────────────────────────────────────
#  Main federated training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_federated(
    X: np.ndarray,
    y: np.ndarray,
    df_raw=None,
) -> dict:
    """
    Run the full federated learning simulation.

    Parameters
    ----------
    X      : scaled feature matrix (all samples)
    y      : binary labels (all samples)
    df_raw : raw dataframe (used for cancer-type partitioning)

    Returns
    -------
    metrics dict from final global evaluation
    """
    device     = torch.device(DEVICE)
    in_features = X.shape[1]

    # ── Hold out global test set ─────────────────────────────────────────
    X_fed, X_test, y_fed, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)

    # ── Partition data across clients ────────────────────────────────────
    logger.info("=== Federated Learning Setup ===")
    logger.info(f"  Partition strategy : {FL['partition']}")

    if FL["partition"] == "cancer_type" and df_raw is not None:
        # Align df_raw to X_fed indices (after train_test_split the indices are original)
        df_fed = df_raw.iloc[:len(X_fed)] if len(df_raw) >= len(X_fed) else df_raw
        partitions = partition_by_cancer_type(X_fed, y_fed, df_fed, FL["num_clients"])
    else:
        partitions = random_partition(X_fed, y_fed, FL["num_clients"])

    clients = [FLClient(i, Xi, yi) for i, (Xi, yi) in enumerate(partitions)]
    logger.info(f"  {len(clients)} clients created")

    # ── Initialize global model ───────────────────────────────────────────
    global_model = LocalMLP(in_features, FL["hidden_dims"], FL["dropout"]).to(device)
    global_state = global_model.state_dict()

    best_auc    = 0.0
    history     = {"round": [], "auc": [], "f1": [], "precision": [], "recall": []}

    logger.info(f"\n=== FL Training ({FL['rounds']} rounds) ===")

    for rnd in range(1, FL["rounds"] + 1):
        # ── Local updates ────────────────────────────────────────────────
        local_states = []
        sample_counts = []
        round_metrics = []

        for client in clients:
            state, metrics = client.local_update(
                global_state, in_features, rnd, device=str(device))
            local_states.append(state)
            sample_counts.append(metrics["n_samples"])
            round_metrics.append(metrics)

        # ── FedAvg aggregation ────────────────────────────────────────────
        global_state = fedavg_aggregate(local_states, sample_counts)

        # ── Global evaluation ────────────────────────────────────────────
        g_metrics = global_evaluate(global_state, X_test, y_test, in_features, str(device))

        history["round"].append(rnd)
        history["auc"].append(g_metrics["auc"])
        history["f1"].append(g_metrics["f1"])
        history["precision"].append(g_metrics["precision"])
        history["recall"].append(g_metrics["recall"])

        if rnd % 5 == 0 or rnd == 1:
            avg_client_auc = np.mean([m["auc"] for m in round_metrics])
            logger.info(
                f"  Round {rnd:3d}/{FL['rounds']} | "
                f"Global AUC={g_metrics['auc']:.4f} | "
                f"Global F1={g_metrics['f1']:.4f} | "
                f"Global Prec={g_metrics['precision']:.4f} | "
                f"Global Recall={g_metrics['recall']:.4f} | "
                f"Avg Client AUC={avg_client_auc:.4f}"
            )

        # Save best
        if g_metrics["auc"] > best_auc:
            best_auc = g_metrics["auc"]
            torch.save(global_state, FL_CKPT)

    logger.info(f"\nBest global AUC: {best_auc:.4f}")
    logger.info(f"FL model saved -> {FL_CKPT}")

    # ── Final evaluation with best model ──────────────────────────────────
    best_state = torch.load(FL_CKPT, weights_only=True)
    final_m    = global_evaluate(best_state, X_test, y_test, in_features, str(device))

    logger.info("\n=== FL FINAL TEST RESULTS ===")
    logger.info(f"  AUC       : {final_m['auc']:.4f}")
    logger.info(f"  F1-score  : {final_m['f1']:.4f}")
    logger.info(f"  Precision : {final_m['precision']:.4f}")
    logger.info(f"  Recall    : {final_m['recall']:.4f}")

    # ── Convergence plot ──────────────────────────────────────────────────
    _plot_fl_convergence(history)

    return final_m


def _plot_fl_convergence(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history["round"], history["auc"], color="#e74c3c",
                 linewidth=2, label="AUC")
    axes[0].set_title("Global Model AUC per FL Round", fontsize=13)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("AUC")
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(history["round"], history["f1"],        label="F1",        color="#2ecc71",  linewidth=2)
    axes[1].plot(history["round"], history["precision"], label="Precision",  color="#3498db",  linewidth=2)
    axes[1].plot(history["round"], history["recall"],    label="Recall",     color="#f39c12",  linewidth=2)
    axes[1].set_title("Global Model Precision / Recall / F1", fontsize=13)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fl_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  FL convergence plot -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from feature_engineering import run_pipeline
    res = run_pipeline()
    run_federated(res["X_scaled"].values, res["y"].values, res["df_raw"])
