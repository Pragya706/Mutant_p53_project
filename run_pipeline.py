"""
run_pipeline.py
===============================================================
ONE FILE TO RUN EVERYTHING — p53 Federated AI Project
===============================================================

Just run this single file in VS Code terminal:
    python run_pipeline.py

Options (edit the CONFIG section below):
    RUN_GNN  = True   -> trains Graph Neural Network
    RUN_FL   = True   -> runs Federated Learning simulation
    CODON    = 175    -> codon for protein target ranking
"""

import os
import sys
import subprocess

# [OK] Proper fix for Windows logging encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except:
    pass

# ================================================================
#  ⚙️  EDIT THESE OPTIONS BEFORE RUNNING
# ================================================================

RUN_GNN  = True   # Set True if you want GNN (needs more time/RAM)
RUN_FL   = True   # Set True for Federated Learning simulation
CODON    = 175     # Mutation codon for protein ranking
                   # Options: 175, 245, 248, 249, 273, 282, 220

# ================================================================
#  DO NOT EDIT BELOW THIS LINE
# ================================================================

import logging
import numpy as np

os.makedirs("results",         exist_ok=True)
os.makedirs("models",          exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("run_pipeline")


def banner(title):
    line = "=" * 60
    log.info(f"\n{line}")
    log.info(f"  {title}")
    log.info(f"{line}")


def check_dataset():
    from config import P53_DATASET
    path = P53_DATASET
    if not os.path.exists(path):
        log.error("Dataset not found!")
        log.error(f"   Expected at: {os.path.abspath(path)}")
        log.error("   Please place tp53_mutations.csv inside data/raw/tp53/")
        sys.exit(1)
    log.info(f"Dataset found: {path}")


def install_check():
    """Warn about missing packages before anything fails."""
    required = {
        "pandas":           "pandas",
        "numpy":            "numpy",
        "sklearn":          "scikit-learn",
        "xgboost":          "xgboost",
        "torch":            "torch",
        "torch_geometric":  "torch_geometric",
        "flwr":             "flwr",
        "networkx":         "networkx",
        "matplotlib":       "matplotlib",
        "seaborn":          "seaborn",
        "joblib":           "joblib",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        log.warning(f"⚠️  Missing packages: {missing}")
        log.warning("   Run:  pip install " + " ".join(missing))
        answer = input("\nInstall missing packages now? (y/n): ").strip().lower()
        if answer == "y":
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install"] + missing)
        else:
            log.error("Please install packages and re-run.")
            sys.exit(1)
    else:
        log.info("[OK]  All packages are installed")


# ── STEP 1 : Load Data ───────────────────────────────────────────────────────
def step1_load():
    banner("STEP 1 — Loading Dataset")
    from src.data_processing.loader import load_raw_csv, basic_info
    df = load_raw_csv()
    basic_info(df)
    log.info(f"[OK]  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ── STEP 2 : Feature Engineering ────────────────────────────────────────────
def step2_features(df):
    banner("STEP 2 — Feature Engineering")
    from src.data_processing.feature_engineering import (
        engineer_features, get_feature_columns)
    df_eng    = engineer_features(df)
    feat_cols = get_feature_columns(df_eng)
    log.info(f"[OK]  Features ready: {len(feat_cols)} numeric feature columns")
    log.info(f"   New columns added: "
             f"{[c for c in df_eng.columns if c not in df.columns]}")
    return df_eng, feat_cols


# ── STEP 3 : Split Data ──────────────────────────────────────────────────────
def step3_split(df_eng):
    banner("STEP 3 — Train / Val / Test Split  (70% / 10% / 20%)")
    from src.data_processing.loader import split_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_eng)
    log.info(f"[OK]  Split done -> "
             f"Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── STEP 4 : Preprocess ──────────────────────────────────────────────────────
def step4_preprocess(X_train, X_val, X_test):
    banner("STEP 4 — Preprocessing  (Impute + Scale + Encode)")
    from src.data_processing.preprocessor import P53Preprocessor
    pp          = P53Preprocessor()
    X_train_p   = pp.fit_transform(X_train)
    X_val_p     = pp.transform(X_val)
    X_test_p    = pp.transform(X_test)
    pp.save()
    log.info(f"[OK]  Preprocessed shape: {X_train_p.shape[1]} features")
    return pp, X_train_p, X_val_p, X_test_p


# ── STEP 5 : PPI Graph ───────────────────────────────────────────────────────
def step5_graph(df_eng):
    banner("STEP 5 — Building PPI Graph  (Protein-Protein Interaction)")
    from src.graph_model.ppi_graph_builder import get_ppi_graph
    G, g_feats, d_feats = get_ppi_graph()
    log.info(f"PPI Graph: {G.number_of_nodes()} nodes, "
             f"{G.number_of_edges()} edges")
    # Return G directly; GNN step uses get_ppi_graph() internally
    return G


# ── STEP 6 : GNN ─────────────────────────────────────────────────────────────
def step6_gnn(X_train_p, y_train_np, X_val_p, y_val_np,
               X_test_p, y_test_np, ppi_graph, num_classes, pp):
    if not RUN_GNN:
        banner("STEP 6 - GNN  [SKIPPED - set RUN_GNN=True to enable]")
        log.info("Using uniform probability fallback for GNN output")
        proba = np.full((len(X_test_p), num_classes), 1.0 / num_classes)
        return None, proba

    banner("STEP 6 - Training Graph Attention Network (GNN)")
    import pandas as pd
    from src.graph_model.gnn_dataset import build_graph_data
    from src.graph_model.train_gnn import train_gnn

    # Reconstruct full scaled DataFrame with correct column names (includes "codon")
    X_all = np.vstack([X_train_p, X_val_p, X_test_p])
    y_all = np.concatenate([y_train_np, y_val_np, y_test_np])
    X_df  = pd.DataFrame(X_all, columns=pp.feature_names)
    y_ser = pd.Series(y_all)

    data = build_graph_data(X_df, y_ser, df_raw=None, cache=True)
    model, test_metrics = train_gnn(data)

    n_test = len(X_test_p)
    raw_p  = test_metrics["probs"]
    p1     = raw_p[:n_test] if len(raw_p) >= n_test else np.resize(raw_p, n_test)
    proba  = np.column_stack([1 - p1, p1])  # shape (n_test, 2)
    log.info(f"GNN training complete - test AUC: {test_metrics['auc']:.4f}")
    return model, proba


# ── STEP 7 : XGBoost + Random Forest ────────────────────────────────────────
def step7_classical(X_train_p, y_train_np, X_val_p, y_val_np, X_test_p):
    banner("STEP 7 — Training XGBoost + Random Forest")
    from src.ml_models.classical_models import XGBoostModel, RandomForestModel

    xgb = XGBoostModel()
    xgb.fit(X_train_p, y_train_np, X_val_p, y_val_np)
    xgb.save()
    xgb_proba = xgb.predict_proba(X_test_p)
    log.info("[OK]  XGBoost done")

    rf = RandomForestModel()
    rf.fit(X_train_p, y_train_np)
    rf.save()
    rf_proba = rf.predict_proba(X_test_p)
    log.info("[OK]  Random Forest done")

    return xgb, rf, xgb_proba, rf_proba


# ── STEP 8 : Federated Learning ──────────────────────────────────────────────
def step8_federated(X_train_p, y_train_np, X_val_p, y_val_np, num_classes):
    if not RUN_FL:
        banner("STEP 8 — Federated Learning  [SKIPPED — set RUN_FL=True to enable]")
        return None

    banner("STEP 8 — Federated Learning Simulation  (5 hospital nodes)")
    from src.federated.fl_server import run_federated
    history = run_federated(
        np.vstack([X_train_p, X_val_p]),
        np.concatenate([y_train_np, y_val_np]),
        df_raw=None)
    log.info("[OK]  Federated Learning simulation complete")
    return history


# ── STEP 9 : Ensemble ────────────────────────────────────────────────────────
def step9_ensemble(gnn_proba, xgb_proba, rf_proba, y_test_np):
    banner("STEP 9 — Ensemble Prediction  (GNN 40% + XGB 35% + RF 25%)")
    from src.ml_models.ensemble_model import EnsemblePredictor
    ensemble  = EnsemblePredictor()
    result    = ensemble.evaluate(gnn_proba, xgb_proba, rf_proba, y_test_np)
    log.info(f"[OK]  Ensemble AUC: {result['auc']}   F1: {result['f1']}")
    return ensemble, result


# ── STEP 10 : Evaluation Metrics ─────────────────────────────────────────────
def step10_evaluate(gnn_proba, xgb_proba, rf_proba,
                     xgb, rf, ens_proba, ens_preds, y_test_np):
    banner("STEP 10 — Evaluation Metrics")
    from src.evaluation.metrics import compute_metrics, compare_models
    results = [
        compute_metrics(y_test_np,
                        (gnn_proba[:, 1] > 0.5).astype(int),
                        gnn_proba[:, 1],
                        "GNN"),

        compute_metrics(y_test_np,
                        xgb_proba.argmax(1),
                        xgb_proba[:, 1],
                        "XGBoost"),

        compute_metrics(y_test_np,
                        rf_proba.argmax(1),
                        rf_proba[:, 1],
                        "Random Forest"),

        compute_metrics(y_test_np,
                        ens_preds,
                        ens_proba[:, 1] if ens_proba.ndim > 1 else ens_proba,
                        "Ensemble"),
    ]
    comp_df = compare_models(results)
    log.info("[OK]  Metrics saved to results/model_comparison.csv")
    return comp_df


# ── STEP 11 : Plots ───────────────────────────────────────────────────────────
def step11_plots(y_test_np, ens_proba, ens_preds,
                  comp_df, pp, xgb_proba, rf_proba):
    banner("STEP 11 — Saving Plots to results/")
    from src.evaluation.metrics import (
        plot_confusion_matrix,
        plot_class_roc,
        plot_feature_importance,
        plot_model_comparison,
        plot_pathogenicity_distribution,
    )
    from src.ml_models.classical_models import XGBoostModel

    xgb_loaded = XGBoostModel.load()
    fi = xgb_loaded.feature_importance(pp.feature_names)

    plot_confusion_matrix(y_test_np, ens_preds)
    plot_class_roc(y_test_np, ens_proba, "Ensemble")
    plot_model_comparison(comp_df)
    plot_pathogenicity_distribution(y_test_np, ens_preds)
    plot_feature_importance(fi, title="XGBoost Feature Importance (Top 25)")

    log.info("[OK]  All plots saved to results/")
    return fi


# ── STEP 12 : Protein Target Ranking ─────────────────────────────────────────
def step12_ranking(df_eng, X_test, ens_proba, fi):
    banner(f"STEP 12 — Protein Target Ranking  (Codon {CODON})")
    from src.ml_models.ensemble_model import (rank_protein_targets,
                                               save_rankings)
    from src.evaluation.metrics import plot_target_ranking

    # Rank for the selected codon
    codon_mask = None
    if "Codon" in X_test.columns:
        codon_mask = (X_test["Codon"] == CODON).values

    if codon_mask is not None and codon_mask.sum() > 0:
        df_codon = df_eng.loc[X_test.index[codon_mask]]
        ranking  = rank_protein_targets(CODON, ens_proba[codon_mask],
                                         df_subset=df_codon,
                                         feature_importance=fi)
    else:
        ranking = rank_protein_targets(CODON, ens_proba,
                                        feature_importance=fi)

    log.info(f"\n{ranking.to_string(index=False)}")
    save_rankings(ranking, CODON)
    plot_target_ranking(ranking, CODON)

    # Generate rankings for ALL hotspot codons
    log.info("Generating rankings for all hotspot codons …")
    # Define hotspot codons locally
    HOTSPOT_CODONS = [175, 245, 248, 249, 273, 282]
    for codon in HOTSPOT_CODONS:
        r = rank_protein_targets(codon, ens_proba, feature_importance=fi)
        save_rankings(r, codon)

    log.info("[OK]  All target rankings saved to results/")


# ================================================================
#  MAIN — runs all steps in order
# ================================================================
def main():
    log.info("")
    log.info("=" * 60)
    log.info("`   Graph-Based Federated AI for Mutant p53 Reactivation   `")
    log.info("`   Group 64 | MSRUAS                                      `")
    log.info("╚==========================================================╝")
    log.info(f"   RUN_GNN = {RUN_GNN}   |   RUN_FL = {RUN_FL}   |   CODON = {CODON}")
    log.info("")

    # Pre-checks
    check_dataset()
    install_check()

    from config import NUM_CLASSES, HOTSPOT_CODONS

    # Run all steps
    df                                            = step1_load()
    df_eng, feat_cols                             = step2_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = step3_split(df_eng)
    pp, X_train_p, X_val_p, X_test_p             = step4_preprocess(
                                                        X_train, X_val, X_test)

    y_train_np = y_train.to_numpy()
    y_val_np   = y_val.to_numpy()
    y_test_np  = y_test.to_numpy()

    ppi_pyg                                       = step5_graph(df_eng)
    gnn_model, gnn_proba                          = step6_gnn(
                                                        X_train_p, y_train_np,
                                                        X_val_p,   y_val_np,
                                                        X_test_p,  y_test_np,
                                                        ppi_pyg,   NUM_CLASSES, pp)
    xgb, rf, xgb_proba, rf_proba                  = step7_classical(
                                                        X_train_p, y_train_np,
                                                        X_val_p,   y_val_np,
                                                        X_test_p)
    step8_federated(X_train_p, y_train_np,
                     X_val_p,   y_val_np, NUM_CLASSES)

    ensemble, ens_result                          = step9_ensemble(
                                                        gnn_proba, xgb_proba,
                                                        rf_proba,  y_test_np)
    ens_proba = ens_result["proba"]
    ens_preds = ens_result["preds"]

    comp_df                                       = step10_evaluate(
                                                        gnn_proba, xgb_proba,
                                                        rf_proba, xgb, rf,
                                                        ens_proba, ens_preds,
                                                        y_test_np)
    fi                                            = step11_plots(
                                                        y_test_np, ens_proba,
                                                        ens_preds, comp_df,
                                                        pp, xgb_proba, rf_proba)
    step12_ranking(df_eng, X_test, ens_proba, fi)

    # ── Final summary ───────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("`                  [OK]  PIPELINE COMPLETE                   `")
    log.info("╠==========================================================╣")
    log.info(f"`  Ensemble AUC  : {ens_result['auc']:<41}`")
    log.info(f"`  Ensemble F1   : {ens_result['f1']:<41}`")
    log.info("╠==========================================================╣")
    log.info("`  Check results/ folder for:                              `")
    log.info("`    • model_comparison.csv   (metrics table)              `")
    log.info("`    • model_comparison.png   (bar chart)                  `")
    log.info("`    • roc_ensemble.png       (ROC curves)                 `")
    log.info("`    • cm_ensemble.png        (confusion matrix)           `")
    log.info("`    • feature_importance.png (top features)               `")
    log.info("`    • target_ranking_*.csv   (protein rankings)           `")
    log.info("`    • pipeline.log           (full log)                   `")
    log.info("╚==========================================================╝")


if __name__ == "__main__":
    main()
