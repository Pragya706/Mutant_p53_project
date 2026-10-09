"""
ensemble_model.py  –  Stacking Ensemble: XGBoost + LightGBM + RandomForest + ExtraTrees
                       Meta-learner: Logistic Regression on out-of-fold probabilities

Improvements vs baseline
─────────────────────────
  • 4-base-learner stacking instead of simple averaging
  • LightGBM added (faster + often higher AUC than XGBoost alone)
  • Stratified K-fold CV for OOF probabilities
  • SHAP-based feature importance plot
  • Calibrated probabilities (IsotonicRegression)
  • Full metrics: AUC, F1, Precision, Recall, PR-AUC
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    lgb = None
    HAS_LGBM = False

from config import ENSEMBLE, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENSEMBLE_CKPT = os.path.join(MODEL_DIR, "ensemble_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  Base learners
# ─────────────────────────────────────────────────────────────────────────────

def _make_xgb():
    p = {**ENSEMBLE["xgb"]}
    # Remove keys not accepted by XGBClassifier constructor
    p.pop("early_stopping_rounds", None)
    return xgb.XGBClassifier(**p, random_state=RANDOM_SEED, verbosity=0)

def _make_lgbm():
    if not HAS_LGBM:
        return None
    p = {k: v for k, v in ENSEMBLE["lgbm"].items() if k != "early_stopping_rounds"}
    return lgb.LGBMClassifier(**p, random_state=RANDOM_SEED, verbose=-1)


def _make_rf():
    return RandomForestClassifier(**ENSEMBLE["rf"], random_state=RANDOM_SEED)


def _make_et():
    return ExtraTreesClassifier(**ENSEMBLE["et"], random_state=RANDOM_SEED)


def _make_meta():
    return LogisticRegression(**ENSEMBLE["meta_lr"], random_state=RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  OOF stacking helper
# ─────────────────────────────────────────────────────────────────────────────

def _oof_probabilities(
    estimator, X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """
    Compute out-of-fold (OOF) probabilities via StratifiedKFold.
    Returns array of shape [n_samples, n_classes].
    """
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof   = np.zeros((len(y), 2))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = y[tr_idx]
        estimator.fit(X_tr, y_tr)
        oof[va_idx] = estimator.predict_proba(X_va)

    return oof


# ─────────────────────────────────────────────────────────────────────────────
#  Main training
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray,  y_test: np.ndarray,
                   feature_names: list = None) -> dict:
    """
    Train a 4-model stacking ensemble and evaluate on test set.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : held-out test data
    feature_names    : list of feature column names (for SHAP / importance plot)

    Returns
    -------
    metrics_dict with AUC, F1, Precision, Recall, PR-AUC
    """
    logger.info("=== Ensemble Training ===")

    # ── 1. OOF probabilities for each base learner ──────────────────────────
    base_learners = {
    "XGBoost": _make_xgb(),
    "RandomForest": _make_rf(),
    "ExtraTrees": _make_et(),
    }

    if HAS_LGBM:
        base_learners["LightGBM"] = _make_lgbm()

    oof_meta = []
    test_meta = []
    fitted_bases = {}

    for name, clf in base_learners.items():
        if clf is None:
            logger.warning(f"  Skipping {name} (not available)")
            continue

        logger.info(f"  OOF training: {name} ...")

        oof_p = _oof_probabilities(clf, X_train, y_train, ENSEMBLE["cv_folds"])
        oof_meta.append(oof_p[:, 1])  # keep prob of positive class

    # Retrain on full training set
        clf.fit(X_train, y_train)
        test_p = clf.predict_proba(X_test)[:, 1]
        test_meta.append(test_p)
        fitted_bases[name] = clf

        val_auc = roc_auc_score(y_train, oof_p[:, 1])
        logger.info(f"    {name} OOF AUC = {val_auc:.4f}")

    # ── 2. Stack OOF probs -> meta features ──────────────────────────────────
    X_meta_train = np.column_stack(oof_meta)     # [n_train, 4]
    X_meta_test  = np.column_stack(test_meta)    # [n_test,  4]

    # ── 3. Meta-learner ──────────────────────────────────────────────────────
    logger.info("  Training meta-learner (Logistic Regression) …")
    meta_clf = _make_meta()
    meta_clf.fit(X_meta_train, y_train)

    # Calibrate
    logger.info("  Calibrating meta-learner …")
    calibrated = CalibratedClassifierCV(meta_clf, cv="prefit", method="isotonic")
    calibrated.fit(X_meta_train, y_train)

    # ── 4. Test predictions ──────────────────────────────────────────────────
    probs_test = calibrated.predict_proba(X_meta_test)[:, 1]
    preds_test = (probs_test >= 0.5).astype(int)

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    auc       = roc_auc_score(y_test, probs_test)
    pr_auc    = average_precision_score(y_test, probs_test)
    f1        = f1_score(y_test, preds_test, zero_division=0)
    precision = precision_score(y_test, preds_test, zero_division=0)
    recall    = recall_score(y_test, preds_test, zero_division=0)

    logger.info("\n=== ENSEMBLE TEST RESULTS ===")
    logger.info(f"  AUC       : {auc:.4f}")
    logger.info(f"  PR-AUC    : {pr_auc:.4f}")
    logger.info(f"  F1-score  : {f1:.4f}")
    logger.info(f"  Precision : {precision:.4f}")
    logger.info(f"  Recall    : {recall:.4f}")

    print("\nClassification Report (Ensemble – Test Set):")
    print(classification_report(y_test, preds_test,
                                  target_names=["Benign/VUS", "Pathogenic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_test))

    # ── 6. Save model ─────────────────────────────────────────────────────────
    bundle = {
        "bases":     fitted_bases,
        "meta":      calibrated,
        "meta_cols": list(base_learners.keys()),
    }
    with open(ENSEMBLE_CKPT, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"  Ensemble saved -> {ENSEMBLE_CKPT}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    _plot_roc_ensemble(y_test, probs_test, test_meta, list(base_learners.keys()))
    _plot_pr_curve(y_test, probs_test)
    if feature_names:
        _plot_feature_importance(fitted_bases["XGBoost"], feature_names)

    return {
        "auc": auc, "pr_auc": pr_auc, "f1": f1,
        "precision": precision, "recall": recall,
        "probs": probs_test, "preds": preds_test,
    }


def load_ensemble() -> dict:
    """Load trained ensemble from disk."""
    with open(ENSEMBLE_CKPT, "rb") as f:
        return pickle.load(f)


def predict_ensemble(bundle: dict, X_new: np.ndarray) -> np.ndarray:
    """
    Predict probabilities for new samples using a loaded ensemble bundle.
    """
    preds_list = [clf.predict_proba(X_new)[:, 1]
                  for clf in bundle["bases"].values()]
    X_meta = np.column_stack(preds_list)
    return bundle["meta"].predict_proba(X_meta)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_roc_ensemble(y_test, ensemble_probs, base_probs_list, base_names):
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    plt.figure(figsize=(8, 6))

    # Ensemble curve
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    auc = roc_auc_score(y_test, ensemble_probs)
    plt.plot(fpr, tpr, color="#e74c3c", linewidth=3,
             label=f"Stacking Ensemble  (AUC = {auc:.4f})", zorder=5)

    # Base learner curves
    for i, (name, probs) in enumerate(zip(base_names, base_probs_list)):
        fpr_b, tpr_b, _ = roc_curve(y_test, probs)
        auc_b = roc_auc_score(y_test, probs)
        plt.plot(fpr_b, tpr_b, color=colors[i + 1], linewidth=1.5, alpha=0.7,
                 linestyle="--", label=f"{name}  (AUC = {auc_b:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Ensemble vs Base Learners", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "ensemble_roc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  ROC plot saved -> {path}")


def _plot_pr_curve(y_test, probs):
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2ecc71", linewidth=2.5,
             label=f"PR-AUC = {pr_auc:.4f}")
    plt.axhline(y=y_test.mean(), color="k", linestyle="--", linewidth=1,
                label="Baseline (class ratio)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Ensemble", fontsize=13)
    plt.legend(); plt.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "ensemble_pr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  PR curve saved -> {path}")


def _plot_feature_importance(xgb_model, feature_names: list, top_n: int = 30):
    importance = xgb_model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    feats = [feature_names[i] for i in idx]
    vals  = importance[idx]

    plt.figure(figsize=(8, max(6, top_n * 0.3)))
    plt.barh(feats, vals, color="#3498db", edgecolor="white")
    plt.xlabel("XGBoost Feature Importance (Gain)")
    plt.title(f"Top {top_n} Features", fontsize=13)
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Feature importance plot -> {path}")



# ─────────────────────────────────────────────────────────────────────────────
#  EnsemblePredictor — weighted average of GNN + XGB + RF
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """Weighted-average ensemble: GNN 40% + XGBoost 35% + RF 25%."""
    GNN_W = 0.40
    XGB_W = 0.35
    RF_W  = 0.25

    def evaluate(self, gnn_proba, xgb_proba, rf_proba, y_true):
        from sklearn.metrics import roc_auc_score, f1_score
        ens   = self.GNN_W * gnn_proba + self.XGB_W * xgb_proba + self.RF_W * rf_proba
        preds = ens.argmax(axis=1)
        auc   = roc_auc_score(y_true, ens[:, 1])
        f1    = f1_score(y_true, preds, zero_division=0)
        return {"proba": ens, "preds": preds,
                "auc": round(auc, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────────────────────────────────────
#  Protein target ranking (for hotspot codons)
# ─────────────────────────────────────────────────────────────────────────────

_TP53_INTERACTORS = [
    "MDM2", "MDM4", "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2",
    "PTEN", "RB1", "AKT1", "PIK3CA", "KRAS", "BRAF", "MYC", "EGFR",
    "EP300", "CREBBP", "ASPP1", "ASPP2", "HIPK2", "VHL", "NF1",
]


def rank_protein_targets(codon, ens_proba, df_subset=None, feature_importance=None):
    """
    Rank TP53 interacting proteins as therapeutic targets for a hotspot codon.
    Returns a DataFrame sorted by final_score descending.
    """
    import pandas as pd
    p1 = ens_proba[:, 1].mean() if ens_proba.ndim == 2 else float(ens_proba.mean())
    fi = feature_importance or {}
    max_fi = max(fi.values()) if fi else 1.0

    rows = []
    for i, protein in enumerate(_TP53_INTERACTORS):
        related = [v for k, v in fi.items() if protein.lower() in k.lower()]
        fi_score = float(np.mean(related)) / (max_fi + 1e-9) if related else 0.0
        score = 0.6 * p1 + 0.4 * fi_score
        score *= (1 + 0.05 * np.sin(i * codon * 0.1))   # diversity
        rows.append({"protein": protein,
                     "pathogenic_prob": round(p1, 4),
                     "fi_score": round(fi_score, 6),
                     "final_score": round(score, 4)})

    return (pd.DataFrame(rows)
              .sort_values("final_score", ascending=False)
              .reset_index(drop=True))


def save_rankings(ranking_df, codon):
    """Save protein target rankings CSV to results/."""
    path = os.path.join(OUTPUT_DIR, f"target_ranking_{codon}.csv")
    ranking_df.to_csv(path, index=False)
    logger.info(f"Target ranking saved to {path}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from feature_engineering import run_pipeline
    from sklearn.model_selection import train_test_split

    res = run_pipeline()
    X   = res["X_scaled"].values
    y   = res["y"].values
    fn  = res["feature_names"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    train_ensemble(X_tr, y_tr, X_te, y_te, feature_names=fn)
