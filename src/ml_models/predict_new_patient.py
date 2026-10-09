"""
Final inference module aligned with training pipeline.
Robust, safe, and Streamlit-ready.
"""

import os
import joblib
import pickle
import pandas as pd
import numpy as np

from src.data_processing.feature_engineering import engineer_features


# ─────────────────────────────────────────────
# REQUIRED INPUT FIELDS
# ─────────────────────────────────────────────
REQUIRED_FIELDS = [
    "Codon",
    "Type",
    "CpG",
    "Domain",
    "WT_AA_1",
    "Mutant_AA_1",
    "Mutation_origin",
]


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def load_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    rf_path  = os.path.join(base_dir, "models", "rf_model.pkl")
    xgb_path = os.path.join(base_dir, "models", "xgboost_model.pkl")

    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb = joblib.load(xgb_path) if os.path.exists(xgb_path) else None

    return rf, xgb


# ─────────────────────────────────────────────
# LOAD PREPROCESSING
# ─────────────────────────────────────────────
def load_preprocessing():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(base_dir, "data", "processed", "features.pkl")

    if not os.path.exists(path):
        return None, None

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None, None

    scaler = data.get("scaler", None)
    feature_names = data.get("feature_names", None)

    return scaler, feature_names


# ─────────────────────────────────────────────
# TARGET RANKING (DEMO)
# ─────────────────────────────────────────────
PROTEIN_TARGETS = {
    "MDM2": "p53 regulator",
    "HSP90": "protein stabilization",
    "ATM": "DNA repair",
    "BRCA1": "tumor suppressor pathway",
    "PIK3CA": "oncogenic signaling",
}

def score_targets(prob):
    return [
        {
            "protein": k,
            "score": round(prob * (1 - i * 0.1), 3),
            "mechanism": v
        }
        for i, (k, v) in enumerate(PROTEIN_TARGETS.items())
    ]


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_patient(patient_dict: dict):

    try:
        # ───────────────── VALIDATION ─────────────────
        missing = [f for f in REQUIRED_FIELDS if f not in patient_dict]
        if missing:
            return {"error": f"Missing required fields: {missing}"}

        # ───────────────── INPUT → DF ─────────────────
        df = pd.DataFrame([patient_dict])

        df["Codon"] = pd.to_numeric(df["Codon"], errors="coerce").fillna(0).astype(int)

        # ───────────────── ADD REQUIRED RAW COLUMNS ─────────────────
        required_defaults = {
            "HG38_Start": 0,
            "Base_Change_Size": 1,
            "HGVS_P": "",
            "HGVS_C": "",
            "cDNA_variant": "",
        }

        for col, val in required_defaults.items():
            if col not in df.columns:
                df[col] = val

        # ───────────────── FEATURE ENGINEERING ─────────────────
        df_eng = engineer_features(df)
        X = df_eng.drop(columns=["Pathogenicity"], errors="ignore")

        # ───────────────── LOAD MODELS ─────────────────
        rf, xgb = load_models()
        if rf is None or xgb is None:
            return {"error": "Model files not found in models/"}

        # ───────────────── LOAD PREPROCESSING ─────────────────
        scaler, feature_names = load_preprocessing()

        # ───────────────── FEATURE ALIGNMENT (CRITICAL FIX) ─────────────────
        if feature_names is not None:
            # Best case: exact training features
            X = X.reindex(columns=feature_names, fill_value=0)
        else:
            # Fallback using model expectation
            # 🔥 SAFE FALLBACK WITHOUT n_features_in_
            try:
                rf.predict(X)
            except Exception as e:
                msg = str(e)

                if "features" in msg:
                    import re
                    match = re.search(r"expecting (\d+) features", msg)
                    if match:
                        expected = int(match.group(1))
                        current = X.shape[1]

                        if current < expected:
                            for i in range(expected - current):
                                X[f"dummy_{i}"] = 0

                        elif current > expected:
                            X = X.iloc[:, :expected]

        # ───────────────── SCALING ─────────────────
        if scaler is not None:
            X = pd.DataFrame(
                scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

        # ───────────────── PREDICTION ─────────────────
        rf_prob  = rf.predict_proba(X)[0][1]
        xgb_prob = xgb.predict_proba(X)[0][1]

        final_prob = 0.6 * xgb_prob + 0.4 * rf_prob

        label = "Pathogenic" if final_prob >= 0.5 else "Benign"

        # ───────────────── OUTPUT ─────────────────
        return {
            "pathogenicity": label,
            "confidence": round(float(final_prob), 3),
            "restoration_score": round(float(final_prob * 0.95), 3),
            "top_targets": score_targets(final_prob),
            "model_breakdown": {
                "xgboost": round(float(xgb_prob), 3),
                "random_forest": round(float(rf_prob), 3),
            }
        }

    except Exception as e:
        return {"error": str(e)}