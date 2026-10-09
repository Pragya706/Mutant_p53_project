"""
tests/test_pipeline.py  –  Unit tests for all three pipeline components
Run with: pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_data():
    """Generate tiny synthetic dataset once for all tests."""
    from tests.generate_synthetic_data import (
        make_tp53_csv, make_string_links, make_string_info, make_biogrid
    )
    make_tp53_csv()
    make_string_links()
    make_string_info()
    make_biogrid()


@pytest.fixture(scope="session")
def pipeline_result(synthetic_data):
    """Run feature engineering once and share across tests."""
    from feature_engineering import run_pipeline
    return run_pipeline(force_rebuild=True)


@pytest.fixture
def small_xy(pipeline_result):
    X = pipeline_result["X_scaled"].values
    y = pipeline_result["y"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  1. Config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_data_paths_are_strings(self):
        from config import P53_DATASET, STRING_LINKS, STRING_PROTEIN_INFO, BIOGRID_FILE
        for path in [P53_DATASET, STRING_LINKS, STRING_PROTEIN_INFO, BIOGRID_FILE]:
            assert isinstance(path, str), f"Path must be a string: {path}"

    def test_output_dirs_defined(self):
        from config import OUTPUT_DIR, MODEL_DIR, LOG_DIR
        for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
            assert isinstance(d, str)

    def test_pathogenicity_map_binary(self):
        from config import PATHOGENICITY_MAP
        assert set(PATHOGENICITY_MAP.values()).issubset({0, 1})

    def test_gnn_config_keys(self):
        from config import GNN
        required = ["hidden_dim", "num_heads", "num_layers", "dropout",
                    "lr", "epochs", "patience"]
        for k in required:
            assert k in GNN, f"GNN config missing key: {k}"

    def test_fl_config_keys(self):
        from config import FL
        required = ["num_clients", "rounds", "local_epochs", "local_lr", "mu"]
        for k in required:
            assert k in FL, f"FL config missing key: {k}"


# ─────────────────────────────────────────────────────────────────────────────
#  2. PPI graph builder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPPIGraph:
    def test_graph_builds(self, synthetic_data):
        from ppi_graph_builder import get_ppi_graph
        G, g_feats, d_feats = get_ppi_graph(force_rebuild=True)
        assert G is not None
        assert G.number_of_nodes() >= 2
        assert G.number_of_edges() >= 1

    def test_tp53_node_present(self, synthetic_data):
        from ppi_graph_builder import get_ppi_graph
        G, _, _ = get_ppi_graph()
        assert "TP53" in G.nodes()

    def test_global_features_keys(self, synthetic_data):
        from ppi_graph_builder import get_ppi_graph
        _, g_feats, _ = get_ppi_graph()
        required = ["ppi_degree", "ppi_weighted_degree", "ppi_driver_score"]
        for k in required:
            assert k in g_feats

    def test_domain_features_nonempty(self, synthetic_data):
        from ppi_graph_builder import get_ppi_graph
        _, _, d_feats = get_ppi_graph()
        assert len(d_feats) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  3. Feature engineering tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_feature_matrix_shape(self, pipeline_result):
        X = pipeline_result["X_scaled"]
        y = pipeline_result["y"]
        assert X.shape[0] == len(y)
        assert X.shape[1] > 10, "Expected more than 10 features"

    def test_no_na_in_features(self, pipeline_result):
        X = pipeline_result["X_scaled"]
        assert not X.isna().any().any(), "Feature matrix contains NaN values"

    def test_no_inf_in_features(self, pipeline_result):
        X = pipeline_result["X_scaled"]
        assert np.all(np.isfinite(X.values)), "Feature matrix contains Inf values"

    def test_target_binary(self, pipeline_result):
        y = pipeline_result["y"]
        assert set(y.unique()).issubset({0, 1}), "Target must be binary (0 or 1)"

    def test_ppi_features_present(self, pipeline_result):
        feature_names = pipeline_result["feature_names"]
        ppi_feats = [f for f in feature_names if f.startswith("ppi_") or f.startswith("dom_ppi_")]
        assert len(ppi_feats) >= 10, f"Expected >=10 PPI features, got {len(ppi_feats)}"

    def test_activity_features_present(self, pipeline_result):
        feature_names = pipeline_result["feature_names"]
        act_feats = [f for f in feature_names if f.startswith("act_")]
        assert len(act_feats) >= 5

    def test_hotspot_feature_present(self, pipeline_result):
        assert "is_hotspot" in pipeline_result["feature_names"]

    def test_codon_feature_present(self, pipeline_result):
        assert "codon" in pipeline_result["feature_names"]


# ─────────────────────────────────────────────────────────────────────────────
#  4. GNN model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNModel:
    def test_forward_pass(self):
        from gnn_model import GATClassifier
        n_nodes, n_feats = 50, 30
        model      = GATClassifier(in_features=n_feats, hidden_dim=32,
                                    num_heads=2, num_layers=2, dropout=0.0)
        x          = torch.randn(n_nodes, n_feats)
        edge_index = torch.randint(0, n_nodes, (2, 100))
        out        = model(x, edge_index)
        assert out.shape == (n_nodes, 2), f"Expected ({n_nodes}, 2), got {out.shape}"

    def test_output_is_logits(self):
        from gnn_model import GATClassifier
        model      = GATClassifier(in_features=20, hidden_dim=32,
                                    num_heads=2, num_layers=2, dropout=0.0)
        x          = torch.randn(20, 20)
        edge_index = torch.randint(0, 20, (2, 40))
        out        = model(x, edge_index)
        # Logits can be any real number; check they're not NaN
        assert not torch.isnan(out).any()

    def test_focal_loss(self):
        from gnn_model import FocalLoss
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        logits  = torch.randn(32, 2)
        targets = torch.randint(0, 2, (32,))
        loss    = loss_fn(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_predict_proba_range(self):
        from gnn_model import GATClassifier
        model      = GATClassifier(in_features=15, hidden_dim=16,
                                    num_heads=2, num_layers=2, dropout=0.0)
        x          = torch.randn(10, 15)
        edge_index = torch.randint(0, 10, (2, 20))
        proba      = model.predict_proba(x, edge_index)
        assert proba.min().item() >= 0.0
        assert proba.max().item() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  5. GNN dataset tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGNNDataset:
    def test_graph_data_builds(self, pipeline_result):
        from gnn_dataset import build_graph_data
        data = build_graph_data(
            pipeline_result["X_scaled"],
            pipeline_result["y"],
            pipeline_result["df_raw"],
            cache=False,
        )
        assert data.x is not None
        assert data.edge_index is not None
        assert data.y is not None

    def test_masks_cover_all_nodes(self, pipeline_result):
        from gnn_dataset import build_graph_data
        data = build_graph_data(
            pipeline_result["X_scaled"],
            pipeline_result["y"],
            pipeline_result["df_raw"],
            cache=False,
        )
        n = data.num_nodes
        covered = (data.train_mask | data.val_mask | data.test_mask).sum().item()
        assert covered == n, "Some nodes are not covered by any mask"

    def test_no_self_loops(self, pipeline_result):
        from gnn_dataset import build_graph_data
        data = build_graph_data(
            pipeline_result["X_scaled"],
            pipeline_result["y"],
            pipeline_result["df_raw"],
            cache=False,
        )
        src, dst = data.edge_index
        self_loops = (src == dst).sum().item()
        assert self_loops == 0, f"Found {self_loops} self-loops in graph"


# ─────────────────────────────────────────────────────────────────────────────
#  6. Ensemble model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsemble:
    def test_ensemble_trains_and_predicts(self, small_xy):
        from ensemble_model import train_ensemble
        from sklearn.model_selection import train_test_split
        X, y = small_xy
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        metrics = train_ensemble(X_tr, y_tr, X_te, y_te)
        assert "auc" in metrics
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_ensemble_metrics_complete(self, small_xy):
        from ensemble_model import train_ensemble
        from sklearn.model_selection import train_test_split
        X, y = small_xy
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        metrics = train_ensemble(X_tr, y_tr, X_te, y_te)
        for key in ["auc", "f1", "precision", "recall", "pr_auc"]:
            assert key in metrics, f"Missing metric: {key}"


# ─────────────────────────────────────────────────────────────────────────────
#  7. Federated Learning tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFederatedLearning:
    def test_fl_client_local_update(self, small_xy):
        from fl_client import fedprox_local_train
        X, y = small_xy
        model = __import__("fl_client").LocalMLP(X.shape[1], [64, 32])
        import copy
        state = copy.deepcopy(model.state_dict())
        new_state, metrics = fedprox_local_train(
            state, X, y, in_features=X.shape[1],
            hidden_dims=[64, 32], local_epochs=2,
            local_lr=1e-3, mu=0.01, batch_size=64,
        )
        assert "auc" in metrics
        assert "final_loss" in metrics
        assert new_state is not None

    def test_random_partition(self, small_xy):
        from fl_server import random_partition
        X, y = small_xy
        parts = random_partition(X, y, n_clients=3)
        assert len(parts) == 3
        total = sum(len(yi) for _, yi in parts)
        assert total == len(y)

    def test_fedavg_aggregate(self):
        from fl_server import fedavg_aggregate
        import torch

        # Two identical state dicts -> aggregation should equal the original
        state1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
        state2 = {"w": torch.tensor([3.0, 2.0, 1.0])}
        aggregated = fedavg_aggregate([state1, state2], weights=[1, 1])
        expected   = torch.tensor([2.0, 2.0, 2.0])
        assert torch.allclose(aggregated["w"], expected), \
            f"FedAvg aggregation wrong: {aggregated['w']} ≠ {expected}"

    def test_fl_runs_few_rounds(self, small_xy, pipeline_result):
        """Minimal end-to-end FL run with just 2 rounds."""
        from config import FL
        import copy
        original_rounds = FL["rounds"]
        FL["rounds"] = 2      # override for speed

        from fl_server import run_federated
        X, y = small_xy
        metrics = run_federated(X, y, df_raw=pipeline_result["df_raw"])

        FL["rounds"] = original_rounds   # restore
        assert 0.0 <= metrics["auc"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  8. Integration: data paths exist after synthetic generation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPaths:
    def test_tp53_file_exists(self, synthetic_data):
        from config import P53_DATASET
        assert os.path.exists(P53_DATASET), f"TP53 dataset not found: {P53_DATASET}"

    def test_string_links_exists(self, synthetic_data):
        from config import STRING_LINKS
        assert os.path.exists(STRING_LINKS), f"STRING links not found: {STRING_LINKS}"

    def test_string_info_exists(self, synthetic_data):
        from config import STRING_PROTEIN_INFO
        assert os.path.exists(STRING_PROTEIN_INFO)

    def test_biogrid_exists(self, synthetic_data):
        from config import BIOGRID_FILE
        assert os.path.exists(BIOGRID_FILE)
