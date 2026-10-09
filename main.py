"""
main.py  –  Full Pipeline Orchestration
          Graph-Based Federated AI for Mutant p53 Reactivation

Run order
──────────
  1. Feature Engineering  (+ PPI graph construction)
  2. GNN (GAT)            –  node-level pathogenicity prediction
  3. Ensemble             –  tabular stacking classifier
  4. Federated Learning   –  privacy-preserving simulation

Usage
─────
  python main.py                   # run all three components
  python main.py --skip-gnn        # skip GNN (faster)
  python main.py --rebuild         # force rebuild of all cached artefacts
  python main.py --component gnn   # run only GNN
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, MODEL_DIR, RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "pipeline.log"), mode="w",
                             encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
os.makedirs("logs", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def step_feature_engineering(rebuild: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 1: Feature Engineering + PPI Graph")
    logger.info("=" * 60)
    from feature_engineering import run_pipeline
    return run_pipeline(force_rebuild=rebuild)


def step_gnn(res: dict, rebuild: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 2: GNN (Graph Attention Network)")
    logger.info("=" * 60)
    from gnn_dataset import build_graph_data
    from train_gnn import train_gnn

    data  = build_graph_data(res["X_scaled"], res["y"], res["df_raw"],
                              cache=not rebuild)
    _, metrics = train_gnn(data)
    return metrics


def step_ensemble(res: dict) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 3: Stacking Ensemble Classifier")
    logger.info("=" * 60)
    from ensemble_model import train_ensemble

    X = res["X_scaled"].values
    y = res["y"].values
    fn = res["feature_names"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    return train_ensemble(X_tr, y_tr, X_te, y_te, feature_names=fn)


def step_fl(res: dict) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 4: Federated Learning (FedProx)")
    logger.info("=" * 60)
    from fl_server import run_federated

    return run_federated(
        res["X_scaled"].values,
        res["y"].values,
        df_raw=res["df_raw"],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(gnn_m: dict | None, ens_m: dict | None, fl_m: dict | None):
    width = 65
    div   = "─" * width

    print(f"\n{'=' * width}")
    print("  FINAL RESULTS SUMMARY")
    print("  Graph-Based Federated AI for Mutant p53 Reactivation")
    print(f"{'=' * width}")

    header = f"{'Component':<22}{'AUC':>8}{'F1':>8}{'Precision':>12}{'Recall':>10}"
    print(header)
    print(div)

    def _row(name, m):
        if m is None:
            return f"  {name:<20}{'—':>8}{'—':>8}{'—':>12}{'—':>10}"
        return (f"  {name:<20}"
                f"{m['auc']:>8.4f}"
                f"{m['f1']:>8.4f}"
                f"{m['precision']:>12.4f}"
                f"{m['recall']:>10.4f}")

    print(_row("GAT (GNN)", gnn_m))
    print(_row("Stacking Ensemble", ens_m))
    print(_row("Federated (FedProx)", fl_m))
    print(div)
    print(f"\nOutputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Models saved to : {os.path.abspath(MODEL_DIR)}")
    print(f"{'=' * width}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the p53 mutation pathogenicity prediction pipeline."
    )
    parser.add_argument("--component", choices=["all", "gnn", "ensemble", "fl"],
                        default="all", help="Which component to run")
    parser.add_argument("--skip-gnn",  action="store_true",
                        help="Skip GNN training (fast mode)")
    parser.add_argument("--rebuild",   action="store_true",
                        help="Force rebuild of cached features & graph data")
    args = parser.parse_args()

    start = time.time()

    # ── Feature engineering (always needed) ─────────────────────────────────
    res = step_feature_engineering(rebuild=args.rebuild)

    gnn_m, ens_m, fl_m = None, None, None

    run_all     = args.component == "all"
    run_gnn     = run_all and not args.skip_gnn
    run_ensemble = run_all or args.component == "ensemble"
    run_fl       = run_all or args.component == "fl"

    if args.component == "gnn":
        run_gnn      = True
        run_ensemble = False
        run_fl       = False

    # ── GNN ──────────────────────────────────────────────────────────────────
    if run_gnn:
        try:
            gnn_m = step_gnn(res, rebuild=args.rebuild)
        except Exception as e:
            logger.error(f"GNN step failed: {e}", exc_info=True)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    if run_ensemble:
        try:
            ens_m = step_ensemble(res)
        except Exception as e:
            logger.error(f"Ensemble step failed: {e}", exc_info=True)

    # ── FL ────────────────────────────────────────────────────────────────────
    if run_fl:
        try:
            fl_m = step_fl(res)
        except Exception as e:
            logger.error(f"FL step failed: {e}", exc_info=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(gnn_m, ens_m, fl_m)
    logger.info(f"Total runtime: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
