

import os

# ─────────────────────────────────────────────
#  ROOT DIRECTORIES
# ─────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "raw")
DATA_RAW   = DATA_DIR                                # alias used by loader.py
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")       # ← your folder is results/
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
LOG_DIR    = os.path.join(ROOT_DIR, "results")       # ← logs go into results/
DATA_PROCESSED  = os.path.join(ROOT_DIR, "data", "processed")

# ─────────────────────────────────────────────
#  DATA PATHS
#
#  data/
#  └── raw/
#      ├── biogrid/
#      │   └── biogrid_all_4.4.243.tab3.txt
#      ├── string/
#      │   ├── string_links_human_v12.txt
#      │   └── 9606_protein_info_v12_0.txt
#      └── tp53/
#          └── tp53_mutations.csv
# ─────────────────────────────────────────────
P53_DATASET         = os.path.join(DATA_DIR, "tp53",    "tp53_mutations.csv")

# Aliases used by loader.py
DATA_RAW     = DATA_DIR
CSV_FILENAME = os.path.join("tp53", "tp53_mutations.csv")

# ─────────────────────────────────────────────
#  LOADER CONSTANTS
# ─────────────────────────────────────────────
TARGET_COLUMN = "Pathogenicity"

# Binary label map  (Pathogenic=1, Benign/VUS=0)
LABEL_MAP = {
    "Pathogenic":          1,
    "Likely Pathogenic":   1,
    "Possibly pathogenic": 1,
    "VUS":                 0,
    "Benign":              0,
    "Likely benign":       0,
}

NUM_CLASSES = 2   # binary classification

# Columns to drop before modelling
# NOTE: activity cols (WAF1_, MDM2_ etc.) and tool score cols are kept here
# because feature_engineering.py reads them directly to build features.
# Only drop pure identifier / free-text / transcript columns.
DROP_COLS = [
    # Identifier columns
    "UMD_ID", "COSMIC_ID", "HG38_Variant",
    "NG_017013.2_Variant",
    "Mutant_Allele", "Wt_Codon", "Mutant_Codon",
    # Free-text columns
    "Variant Comment", "Disease",
    # Unnamed/empty column
    "Unnamed: 34",
    # Transcript columns (not used in pipeline)
    "Transcript t1", "Transcript t2", "Transcript t3",
    "Transcript t4", "Transcript t5", "Transcript t6",
    "Transcript t7", "Transcript t8",
    # Extra columns not needed
    "Variant_Type", "Sample_pathology", "Sample_origin",
    "Internal", "Solid", "Complexity",
    "Variant_Classification", "Prediction_Label",
]

# Columns stored as strings that should be cast to numeric
STR_TO_NUMERIC_COLS = [
    "Sift Score", "Provean_Score", "Mutassessor_score",
    "Polyphen", "Condel", "MutPred_Splice_General_Score",
]

# Train / val / test split ratios
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10
RANDOM_STATE = 42

# Hotspot codons for protein target ranking
HOTSPOT_CODONS = [175, 220, 245, 248, 249, 273, 282]
STRING_LINKS        = os.path.join(DATA_DIR, "string",  "string_links_human_v12.txt")
STRING_PROTEIN_INFO = os.path.join(DATA_DIR, "string",  "9606_protein_info_v12_0.txt")
BIOGRID_FILE        = os.path.join(DATA_DIR, "biogrid", "biogrid_all_4.4.243.tab3.txt")

# Processed / cached artefacts  (all go into results/)
PPI_GRAPH_CACHE  = os.path.join(OUTPUT_DIR, "ppi_graph.pkl")
FEATURES_CACHE   = os.path.join(OUTPUT_DIR, "features_engineered.pkl")
GRAPH_DATA_CACHE = os.path.join(OUTPUT_DIR, "gnn_data.pt")

# Saved model checkpoints  (all go into models/)
GNN_CKPT      = os.path.join(MODEL_DIR, "best_gnn.pt")
FL_CKPT       = os.path.join(MODEL_DIR, "best_fl_model.pt")
ENSEMBLE_CKPT = os.path.join(MODEL_DIR, "ensemble_model.pkl")

# ─────────────────────────────────────────────
#  LOADER CONSTANTS  (used by src/data_processing/loader.py)
# ─────────────────────────────────────────────
CSV_FILENAME   = os.path.join("tp53", "tp53_mutations.csv")
TARGET_COLUMN  = "Pathogenicity"

# Columns to drop before modelling (IDs, free-text, redundant)
DROP_COLS = [
    "Database_ID", "UMD_ID", "COSMIC_ID",
    "HG38_Variant", "NG_017013.2_Variant",
    "cDNA_variant", "Mutant_Allele",
    "Wt_Codon", "Mutant_Codon",
    "Variant Comment", "Variant_Classification",
    "Variant_Type", "Mutation_origin",
    "Disease", "Sample_pathology", "Sample_origin",
    "Internal", "Solid", "Complexity",
    "Transcript t1", "Transcript t2", "Transcript t3",
    "Transcript t4", "Transcript t5", "Transcript t6",
    "Transcript t7", "Transcript t8",
    "Prediction_Label",                  # direct leakage of target
]

# Map Pathogenicity string -> integer label
LABEL_MAP = {
    "Pathogenic":          1,
    "Likely Pathogenic":   1,
    "Possibly pathogenic": 1,
    "VUS":                 0,
    "Benign":              0,
}

# String columns that should be treated as numeric
STR_TO_NUMERIC_COLS = [
    "Sift Score", "Provean_Score", "Mutassessor_score",
    "Polyphen", "Condel", "MutPred_Splice_General_Score",
    "PCA_Score",
    "WAF1_", "MDM2_", "BAX_", "__14_3_3_s",
    "AIP_", "GADD45_", "NOXA_", "p53R2_",
    "Leukemia_Stat", "Solid_Stat", "Tumor_Stat",
    "Cell_line_Stat", "Somatic_Stat", "Germline_Stat",
]

# Train / val / test split sizes
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 20 % held-out test
VAL_SIZE     = 0.10   # 10 % validation  (taken from train portion)

# Number of target classes (Pathogenic=1 vs Benign/VUS=0)
NUM_CLASSES  = 2

# Hotspot codons for protein target ranking
HOTSPOT_CODONS = [175, 220, 245, 248, 249, 273, 282]

# ─────────────────────────────────────────────
#  TP53 IDENTIFIERS
# ─────────────────────────────────────────────
TP53_STRING_ID   = "9606.ENSP00000269305"
TP53_GENE_SYMBOL = "TP53"
TP53_ENTREZ_ID   = 7157
HUMAN_TAXON_ID   = 9606

# ─────────────────────────────────────────────
#  TP53 DOMAIN BOUNDARIES  (UniProt P04637)
# ─────────────────────────────────────────────
TP53_DOMAINS = {
    "Transactivation TAD1":   (1,   40,  "MDM2/MDM4_TAD"),
    "Transactivation TAD2":   (40,  67,  "MDM2/RNAPII_TAD2"),
    "Proline Rich":           (67,  98,  "PXXP_SH3"),
    "DNA binding":            (102, 292, "DNA/ASPP/53BP1"),
    "HCD II - DNA binding":   (102, 145, "DNA/ASPP/53BP1"),
    "HCD III - DNA binding":  (146, 200, "DNA/ASPP/53BP1"),
    "HCD IV - DNA binding":   (200, 248, "DNA/MDM2"),
    "HCD V - DNA binding":    (248, 292, "DNA/BRCA1"),
    "Linker":                 (293, 325, "NLS_14-3-3"),
    "NLS":                    (305, 322, "Importin"),
    "Oligomerization":        (323, 356, "Tetramer_MDM2"),
    "NES - Oligomerization":  (339, 352, "CRM1_Exportin"),
    "Regulatory":             (356, 393, "14-3-3/CBP/p300"),
}

TP53_CANCER_DRIVER_INTERACTORS = {
    "MDM2", "MDM4", "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2",
    "PTEN", "RB1", "CDKN2A", "CDK4", "CDK6", "SMAD2", "SMAD3", "SMAD4",
    "AKT1", "PIK3CA", "KRAS", "BRAF", "MYC", "MYCN", "EGFR", "ALK",
    "EP300", "CREBBP", "ASPP1", "ASPP2", "HIPK2", "COP1", "PIRH2",
    "VHL", "NF1", "NF2", "TSC1", "TSC2", "STK11",
}

STRING_SCORE_THRESHOLD = 400

BIOGRID_PHYSICAL_SYSTEMS = {
    "Two-hybrid", "Affinity Capture-MS", "Affinity Capture-Western",
    "Co-crystal Structure", "Protein-peptide", "Biochemical Activity",
    "Reconstituted Complex", "FRET", "PCA", "Co-purification",
    "Far Western", "Affinity Capture-RNA", "Co-fractionation",
}

# ─────────────────────────────────────────────
#  TARGET DEFINITION  (alias kept for backward compatibility)
# ─────────────────────────────────────────────
PATHOGENICITY_MAP = LABEL_MAP

# ─────────────────────────────────────────────
#  GNN HYPERPARAMETERS
# ─────────────────────────────────────────────
GNN = dict(
    hidden_dim       = 256,
    num_heads        = 4,
    num_layers       = 3,
    dropout          = 0.3,
    edge_codon_range = 3,
    same_domain_edge = False,   
    lr               = 1e-3,
    weight_decay     = 1e-4,
    epochs           = 200,
    patience         = 20,
    batch_size       = 2048,
    train_val_test   = (0.7, 0.15, 0.15),
)

# ─────────────────────────────────────────────
#  ENSEMBLE HYPERPARAMETERS
# ─────────────────────────────────────────────
ENSEMBLE = dict(
    cv_folds = 5,
    seed     = 42,
    xgb  = dict(n_estimators=1000, max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="auc", early_stopping_rounds=50, n_jobs=-1),
    rf   = dict(n_estimators=500, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, class_weight="balanced"),
    lgbm = dict(n_estimators=1000, max_depth=7, learning_rate=0.05,
                num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, class_weight="balanced"),
    et   = dict(n_estimators=500, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, class_weight="balanced"),
    meta_lr = dict(C=1.0, max_iter=1000),
)

# ─────────────────────────────────────────────
#  FEDERATED LEARNING
# ─────────────────────────────────────────────
FL = dict(
    num_clients  = 5,
    rounds       = 50,
    local_epochs = 5,
    local_lr     = 1e-3,
    mu           = 0.01,
    partition    = "cancer_type",
    hidden_dims  = [256, 128, 64],
    dropout      = 0.3,
    batch_size   = 256,
    seed         = 42,
)

# ─────────────────────────────────────────────
#  MISC
# ─────────────────────────────────────────────
RANDOM_SEED = 42
N_JOBS      = -1

# ─────────────────────────────────────────────
#  ALIASES & EXTRA CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR  = MODEL_DIR   # classical_models.py uses MODELS_DIR
RESULTS_DIR = OUTPUT_DIR  # metrics.py uses RESULTS_DIR

# Label order for binary classification
LABEL_ORDER = ["Benign/VUS", "Pathogenic"]

# XGBoost params for classical_models.py
XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", n_jobs=-1,
)

# Random Forest params for classical_models.py
RF_PARAMS = dict(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, class_weight="balanced",
)

DEVICE      = "cpu"   # change to "cuda" if GPU available
