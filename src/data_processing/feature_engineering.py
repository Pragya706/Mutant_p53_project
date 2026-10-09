"""
feature_engineering.py – Clean, leakage-free, pipeline-compatible version
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from config import (
    P53_DATASET, FEATURES_CACHE, OUTPUT_DIR,
    PATHOGENICITY_MAP, TP53_DOMAINS,
)

import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
from src.graph_model.ppi_graph_builder import get_ppi_graph

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def safe_float(series):
    return pd.to_numeric(series, errors="coerce")


def encode_amino_acid(aa1):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY") + ["*", "X"]
    return {f"aa_{a}": int(aa1 == a) for a in aa_list}


def nucleotide_features(cDNA_variant):
    feats = {f"nuc_{b}_{t}": 0 for b in "ACGT" for t in ["ref", "alt"]}
    try:
        s = str(cDNA_variant)
        idx = s.find(">")
        if idx > 0:
            ref = s[idx - 1].upper()
            alt = s[idx + 1].upper()
            feats[f"nuc_{ref}_ref"] = 1
            feats[f"nuc_{alt}_alt"] = 1
    except:
        pass
    return feats


def codon_to_domain(codon):
    for name, (start, end, _) in TP53_DOMAINS.items():
        if start <= codon <= end:
            return name
    return "Unknown"


# ─────────────────────────────────────────────
# Feature Builder (SAFE)
# ─────────────────────────────────────────────

def build_features(df, domain_ppi_feats, global_ppi_feats,
                   fit_encoders=True, encoders=None):

    if encoders is None:
        encoders = {}

    feat = pd.DataFrame(index=df.index)

    # BASIC
    feat["codon"] = safe_float(df["Codon"]).fillna(0).astype(int)
    feat["hg38_start"] = safe_float(df["HG38_Start"]).fillna(0)
    feat["base_change_size"] = safe_float(df["Base_Change_Size"]).fillna(0)

    feat["domain_mapped"] = feat["codon"].apply(codon_to_domain)

    # AMINO ACID
    wt = df["WT_AA_1"].fillna("X").str.upper().str[0]
    mut = df["Mutant_AA_1"].fillna("X").str.upper().str[0]

    wt_ohe = pd.DataFrame([encode_amino_acid(a) for a in wt], index=df.index)
    mut_ohe = pd.DataFrame([encode_amino_acid(a) for a in mut], index=df.index)

    wt_ohe.columns = [c.replace("aa_", "wt_aa_") for c in wt_ohe.columns]
    mut_ohe.columns = [c.replace("aa_", "mut_aa_") for c in mut_ohe.columns]

    feat = pd.concat([feat, wt_ohe, mut_ohe], axis=1)

    feat["is_nonsense"] = (mut == "*").astype(int)
    feat["is_synonymous"] = (wt == mut).astype(int)

    # NUCLEOTIDE
    # Safe handling of cDNA_variant column
    if "cDNA_variant" in df.columns:
        cDNA_series = df["cDNA_variant"].fillna("")
    else:
        cDNA_series = pd.Series([""] * len(df), index=df.index)

    nuc = pd.DataFrame(
        [nucleotide_features(v) for v in cDNA_series],
        index=df.index
    )
    feat = pd.concat([feat, nuc], axis=1)

    # MUTATION TYPE
    feat["is_transition"] = (df.get("Type", "").fillna("").str.upper() == "TS").astype(int)
    feat["is_cpg"] = (df.get("CpG", "").fillna("no").str.lower() == "yes").astype(int)

    if "Mutation_origin" in df.columns:
        origin = df["Mutation_origin"].fillna("").str.lower()
    else:
        origin = pd.Series([""] * len(df), index=df.index)
    feat["is_somatic"] = (origin == "somatic").astype(int)
    feat["is_germline"] = (origin == "germline").astype(int)

    # SAFE CATEGORICAL (keep naming consistent)
    cat_map = {
        "Domain": "domain_enc",
        "Variant_Classification": "variant_class_enc",
        "Mutational_event": "mut_event_enc",
    }

    for raw, new in cat_map.items():
        if raw not in df.columns:
            feat[new] = 0
            continue

        series = df[raw].fillna("Unknown").str.upper()

        if fit_encoders:
            le = LabelEncoder()
            feat[new] = le.fit_transform(series)
            encoders[new] = le
        else:
            le = encoders.get(new)
            feat[new] = le.transform(series) if le else 0

    # HOTSPOT
    hotspot = {175, 245, 248, 249, 273, 282}
    feat["is_hotspot"] = feat["codon"].isin(hotspot).astype(int)

    feat["codon_frequency"] = np.log1p(
        feat["codon"].map(feat["codon"].value_counts())
    )

    if "Tumor_Repetition" in df.columns:
        feat["tumor_repetition"] = np.log1p(
            safe_float(df["Tumor_Repetition"]).fillna(0)
        )

    # PPI FEATURES
    feat["ppi_degree"] = global_ppi_feats.get("ppi_degree", 0)
    feat["ppi_weighted_degree"] = global_ppi_feats.get("ppi_weighted_degree", 0)
    feat["ppi_betweenness"] = global_ppi_feats.get("ppi_betweenness", 0)

    for f in ["n_domain_partners", "mean_score"]:
        feat[f"dom_ppi_{f}"] = feat["domain_mapped"].map(
            lambda d: domain_ppi_feats.get(d, {}).get(f, 0)
        )

    feat.drop(columns=["domain_mapped"], inplace=True)

    # CLEAN
    feat = feat.fillna(0)

    logger.info(f"Feature matrix shape: {feat.shape}")

    return feat, encoders


# ─────────────────────────────────────────────
# Target
# ─────────────────────────────────────────────

def build_target(df):
    return df["Pathogenicity"].map(PATHOGENICITY_MAP).fillna(0).astype(int)


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(force_rebuild=False):

    if not force_rebuild and os.path.exists(FEATURES_CACHE):
        with open(FEATURES_CACHE, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(P53_DATASET, low_memory=False)

    G, g_feats, d_feats = get_ppi_graph()

    X, encoders = build_features(df, d_feats, g_feats)

    y = build_target(df)

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    result = {
        "X_raw": X,
        "X_scaled": X_scaled,
        "y": y,
        "feature_names": list(X.columns),
        "encoders": encoders,
        "scaler": scaler,
        "df_raw": df
    }

    with open(FEATURES_CACHE, "wb") as f:
        pickle.dump(result, f)

    return result


# ─────────────────────────────────────────────
# Wrappers (IMPORTANT for pipeline)
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        _G, g_feats, d_feats = get_ppi_graph()
    except:
        g_feats, d_feats = {}, {}

    feat, _ = build_features(df, d_feats, g_feats)

    if "Pathogenicity" in df.columns:
        feat["Pathogenicity"] = build_target(df).values

    return feat


def get_feature_columns(df_eng: pd.DataFrame) -> list:
    exclude = {"Pathogenicity"}
    return [c for c in df_eng.columns if c not in exclude]