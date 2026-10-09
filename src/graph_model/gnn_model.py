"""
gnn_model.py  –  Graph Attention Network (GAT) for TP53 mutation pathogenicity

Architecture
────────────
  Input  -> GATConv(hidden) + BN + ReLU + Dropout
         -> GATConv(hidden) + BN + ReLU + Dropout  
         -> GATConv(hidden) + BN + ReLU + Dropout  
         -> Linear classifier head
         -> Softmax (2 classes) / Sigmoid (binary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool


class GATBlock(nn.Module):
    """Single GAT layer with BatchNorm and residual connection."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.conv    = GATConv(in_dim, out_dim // heads, heads=heads,
                               dropout=dropout, concat=True)
        self.bn      = BatchNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip    = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.elu(out + self.skip(x))
        out = self.dropout(out)
        return out


class GATClassifier(nn.Module):
    """
    Multi-layer GAT node classifier with:
      - Skip / residual connections between layers
      - Batch normalisation after each GAT block
      - Auxiliary MLP path (feeds raw features to final classifier for ensemble effect)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes  = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # GAT blocks
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATBlock(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
            )

        # Auxiliary MLP (raw features -> skip path to classifier)
        self.aux_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ELU(),
        )

        # Classifier head
        combined_dim = hidden_dim + hidden_dim // 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch=None):
        # Auxiliary raw path
        aux = self.aux_mlp(x)                       

        # Graph path
        h = self.input_proj(x)                     
        for gat in self.gat_layers:
            h = gat(h, edge_index)                  

        # Concatenate graph embedding + raw features
        combined = torch.cat([h, aux], dim=1)       

        out = self.classifier(combined)             
        return out

    def predict_proba(self, x, edge_index):
        """Return softmax probability of positive class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=1)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss (handles class imbalance better than cross-entropy)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pt    = probs[range(len(targets)), targets]
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        loss  = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    n_nodes, n_feats = 100, 80
    x          = torch.randn(n_nodes, n_feats)
    edge_index = torch.randint(0, n_nodes, (2, 500))
    model      = GATClassifier(in_features=n_feats, hidden_dim=64, num_heads=4,
                                num_layers=3, dropout=0.3)
    out = model(x, edge_index)
    print(f"Output shape: {out.shape}")          # [100, 2]
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
