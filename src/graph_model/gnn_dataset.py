"""
gnn_dataset.py  –  Build PyTorch Geometric Data / Dataset from mutation features + PPI graph.

Graph construction strategy
────────────────────────────
  Nodes  : every TP53 mutation (one row = one node)
  Edges  : two mutations are connected if
             a) they share the same structural domain  (same-domain edge)
             b) their codon positions are within ± CODON_RANGE  (proximity edge)
  Features : engineered feature vector (from feature_engineering.py)
  Labels   : binary pathogenicity
"""

import os
import logging
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import train_test_split

from config import GRAPH_DATA_CACHE, GNN, RANDOM_SEED, OUTPUT_DIR, TP53_DOMAINS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Edge builders
# ─────────────────────────────────────────────────────────────────────────────

def _codon_proximity_edges(codons: np.ndarray, codon_range: int = 5) -> torch.Tensor:

    n = len(codons)
    order = np.argsort(codons)
    sorted_c = codons[order]

    src, dst = [], []

    MAX_NEIGHBORS = 10   # 🔥 CRITICAL LIMIT

    lo = 0
    for hi in range(n):
        while sorted_c[hi] - sorted_c[lo] > codon_range:
            lo += 1

        count = 0
        for j in range(hi - 1, lo - 1, -1):  # go backwards (closer neighbors first)
            src.append(order[j])
            dst.append(order[hi])
            src.append(order[hi])
            dst.append(order[j])

            count += 1
            if count >= MAX_NEIGHBORS:
                break   # 🔥 LIMIT EDGES

    return torch.tensor([src, dst], dtype=torch.long)


def _same_domain_edges(domain_ids: np.ndarray) -> torch.Tensor:
    """
    Connect all pairs of mutations in the same domain.
    Limits to domains with ≤ MAX_DOMAIN_SIZE mutations to avoid quadratic blowup.
    """
    MAX_DOMAIN_SIZE = 2000  # safety cap

    src, dst = [], []
    unique_domains = np.unique(domain_ids)
    for d in unique_domains:
        idx = np.where(domain_ids == d)[0]
        if len(idx) > MAX_DOMAIN_SIZE:
            # random subsample for very large domains (DBD has ~42k mutations)
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(idx, MAX_DOMAIN_SIZE, replace=False)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                src += [idx[i], idx[j]]
                dst += [idx[j], idx[i]]

    return torch.tensor([src, dst], dtype=torch.long)


def build_graph_data(X_scaled, y, df_raw=None, cache: bool = True) -> Data:
    """
    Build a single large PyG Data object from the mutation feature matrix.

    Parameters
    ----------
    X_scaled : pd.DataFrame  –  scaled feature matrix (n_mutations × n_features)
    y        : pd.Series     –  binary labels
    df_raw   : pd.DataFrame  –  raw df (used to extract codon & domain columns)
    cache    : bool          –  save/load from GRAPH_DATA_CACHE

    Returns
    -------
    data : torch_geometric.data.Data
    """
    if cache and os.path.exists(GRAPH_DATA_CACHE):
        logger.info(f"Loading cached graph data from {GRAPH_DATA_CACHE}")
        return torch.load(GRAPH_DATA_CACHE, weights_only=False)

    logger.info("Building PyG graph …")
    n = len(X_scaled)

    x = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # ── Codon proximity edges ────────────────────────────────────────────────
    codon_col = "codon" if "codon" in X_scaled.columns else None
    if codon_col:
        codons = X_scaled[codon_col].values.astype(int)
    elif df_raw is not None and "Codon" in df_raw.columns:
        codons = df_raw["Codon"].fillna(0).astype(int).values
    else:
        codons = np.zeros(n, dtype=int)

    logger.info("  Building codon proximity edges …")
    edge_proximity = _codon_proximity_edges(codons, GNN["edge_codon_range"])
    logger.info(f"    Proximity edges: {edge_proximity.shape[1] // 2:,}")

    # ── Same-domain edges ────────────────────────────────────────────────────
    if GNN["same_domain_edge"]:
        # Map each mutation to a domain ID
        domain_id = np.zeros(n, dtype=int)
        domain_list = list(TP53_DOMAINS.keys())
        domain_name_to_id = {d: i for i, d in enumerate(domain_list)}
        n_unknown = len(TP53_DOMAINS)

        if df_raw is not None and "Domain" in df_raw.columns:
            for i, d in enumerate(df_raw["Domain"].fillna("Unknown").values):
                domain_id[i] = domain_name_to_id.get(str(d).strip(), n_unknown)
        else:
            # derive from codon
            def _codon_to_id(c):
                for did, (dname, (s, e, _)) in enumerate(TP53_DOMAINS.items()):
                    if s <= c <= e:
                        return did
                return n_unknown
            domain_id = np.array([_codon_to_id(c) for c in codons])

        logger.info("  Building same-domain edges …")
        edge_domain = _same_domain_edges(domain_id)
        logger.info(f"    Domain edges:    {edge_domain.shape[1] // 2:,}")

        if edge_domain.shape[1] > 0 and edge_proximity.shape[1] > 0:
            edge_index = torch.cat([edge_proximity, edge_domain], dim=1)
        elif edge_domain.shape[1] > 0:
            edge_index = edge_domain
        else:
            edge_index = edge_proximity
    else:
        edge_index = edge_proximity

    # Remove self-loops and duplicate edges
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    edge_index = torch.unique(edge_index, dim=1)

    logger.info(f"  Final edges (undirected pairs): {edge_index.shape[1] // 2:,}")

    # ── Train / val / test masks ─────────────────────────────────────────────
    indices = np.arange(n)
    tr_ratio, val_ratio, _te_ratio = GNN["train_val_test"]
    i_train, i_tmp = train_test_split(indices, test_size=1 - tr_ratio,
                                       random_state=RANDOM_SEED, stratify=y.values)
    val_frac = val_ratio / (1 - tr_ratio)
    i_val, i_test = train_test_split(i_tmp, test_size=1 - val_frac,
                                      random_state=RANDOM_SEED,
                                      stratify=y.values[i_tmp])

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[i_train] = True
    val_mask[i_val]     = True
    test_mask[i_test]   = True

    data = Data(
        x          = x,
        edge_index = edge_index,
        y          = y_tensor,
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
        num_nodes  = n,
    )

    logger.info(f"  Graph summary: nodes={data.num_nodes:,}, edges={data.num_edges:,}, "
                f"features={data.x.shape[1]}, "
                f"train={train_mask.sum().item():,}, "
                f"val={val_mask.sum().item():,}, "
                f"test={test_mask.sum().item():,}")

    if cache:
        torch.save(data, GRAPH_DATA_CACHE)
        logger.info(f"  Graph data cached -> {GRAPH_DATA_CACHE}")

    return data


if __name__ == "__main__":
    from feature_engineering import run_pipeline
    res  = run_pipeline()
    data = build_graph_data(res["X_scaled"], res["y"], res["df_raw"])
    print(data)
