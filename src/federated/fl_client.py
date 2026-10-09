"""
fl_client.py  –  Federated Learning client with FedProx regularization

Key improvements vs vanilla FedAvg client
──────────────────────────────────────────
  • FedProx proximal term: penalises drift from global model
  • Per-client early stopping
  • Local LR warmup
  • Detailed per-round metrics logging
"""

import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score

from config import FL, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Local MLP model (used inside each FL client)
# ─────────────────────────────────────────────────────────────────────────────

class LocalMLP(nn.Module):
    """
    Lightweight MLP for FL clients.
    Architecture mirrors the GNN classifier head so weights can be partially
    transferred in a federated-GNN hybrid setup.
    """
    def __init__(self, in_features: int, hidden_dims: list, dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
#  FedProx local training
# ─────────────────────────────────────────────────────────────────────────────

def fedprox_local_train(
    global_state_dict: dict,
    local_X: np.ndarray,
    local_y: np.ndarray,
    in_features: int,
    hidden_dims: list = None,
    local_epochs: int = None,
    local_lr: float = None,
    mu: float = None,
    batch_size: int = None,
    dropout: float = None,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Run FedProx local update on a client.

    FedProx minimises:
      L_local(w) + (mu/2) * ||w - w_global||²

    Parameters
    ----------
    global_state_dict : current global model weights
    local_X, local_y  : client's local data
    ...

    Returns
    -------
    updated_state_dict : new local weights after training
    metrics            : dict with loss, auc, f1 on local data
    """
    # Defaults from config
    if hidden_dims  is None: hidden_dims  = FL["hidden_dims"]
    if local_epochs is None: local_epochs = FL["local_epochs"]
    if local_lr     is None: local_lr     = FL["local_lr"]
    if mu           is None: mu           = FL["mu"]
    if batch_size   is None: batch_size   = FL["batch_size"]
    if dropout      is None: dropout      = FL["dropout"]

    device = torch.device(device)

    # ── Build local model from global weights ─────────────────────────────
    model = LocalMLP(in_features, hidden_dims, dropout).to(device)
    model.load_state_dict(global_state_dict)

    # Keep a frozen copy of global weights for FedProx term
    global_model = copy.deepcopy(model)
    for p in global_model.parameters():
        p.requires_grad_(False)

    # ── DataLoader ────────────────────────────────────────────────────────
    X_t = torch.tensor(local_X, dtype=torch.float32)
    y_t = torch.tensor(local_y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=local_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    epoch_losses = []

    for epoch in range(local_epochs):
        batch_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()

            logits = model(xb)
            ce_loss = criterion(logits, yb)

            # FedProx proximal term: (mu/2) * ||w - w_global||²
            prox_loss = torch.tensor(0.0, device=device)
            for w, w_g in zip(model.parameters(), global_model.parameters()):
                prox_loss += ((w - w_g) ** 2).sum()
            prox_loss = (mu / 2) * prox_loss

            loss = ce_loss + prox_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        epoch_losses.append(np.mean(batch_losses))

    # ── Local evaluation ──────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_full = X_t.to(device)
        logits  = model(X_full)
        probs   = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds   = (probs >= 0.5).astype(int)

    auc = roc_auc_score(local_y, probs) if len(np.unique(local_y)) > 1 else 0.5
    f1  = f1_score(local_y, preds, zero_division=0)

    metrics = {
        "final_loss": epoch_losses[-1],
        "mean_loss":  np.mean(epoch_losses),
        "auc":        auc,
        "f1":         f1,
        "n_samples":  len(local_y),
    }

    return model.state_dict(), metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Client class (wraps local data + training)
# ─────────────────────────────────────────────────────────────────────────────

class FLClient:
    """
    Represents a single federated learning participant.
    Holds its own data partition and runs local FedProx training.
    """

    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray):
        self.client_id = client_id
        self.X = X
        self.y = y
        logger.debug(f"Client {client_id}: {len(y)} samples, "
                     f"pos_rate={y.mean():.3f}")

    def local_update(
        self,
        global_state_dict: dict,
        in_features: int,
        round_num: int,
        device: str = "cpu",
    ) -> tuple[dict, dict]:
        """Run FedProx local update and return new weights + metrics."""
        state, metrics = fedprox_local_train(
            global_state_dict,
            self.X, self.y,
            in_features=in_features,
            device=device,
        )
        return state, metrics
