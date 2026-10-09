import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    DATA_RAW, CSV_FILENAME, TARGET_COLUMN,
    DROP_COLS, LABEL_MAP, RANDOM_STATE,
    TEST_SIZE, VAL_SIZE, STR_TO_NUMERIC_COLS
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
def load_raw_csv(filename: str = CSV_FILENAME) -> pd.DataFrame:
    path = os.path.join(DATA_RAW, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Place p53_dataset.csv in data/raw/"
        )

    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    # Drop unnecessary columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns],
            inplace=True, errors="ignore")

    # Convert selected columns to numeric
    for col in STR_TO_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"After cleaning: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────
# Debug info (optional)
# ─────────────────────────────────────────────
def basic_info(df: pd.DataFrame) -> None:
    log.info(f"Columns ({len(df.columns)}): {list(df.columns)}")
    log.info(f"Missing values:\n"
             f"{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")

    if TARGET_COLUMN in df.columns:
        log.info(f"Target distribution:\n{df[TARGET_COLUMN].value_counts()}")


# ─────────────────────────────────────────────
# Split data (LEAKAGE-FREE)
# ─────────────────────────────────────────────
def split_data(df: pd.DataFrame):

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"'{TARGET_COLUMN}' not found in dataframe.")

    # Use engineered feature column
    if "codon" not in df.columns:
        raise KeyError("codon column required for GroupSplit")

    groups = df["codon"].fillna(-1).astype(int)

    df = df.copy()

    # Convert target to numeric if needed
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_MAP)

    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 🔥 BETTER GROUPING (codon + amino acid context)
    # ✅ Use engineered features (correct columns)
    if "codon" not in df.columns:
        raise KeyError("codon column required for GroupSplit")

    groups = df["codon"].fillna(-1).astype(int)

    # Reset index for safety
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    # ── Train-Test Split ─────────────────────────
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ── Train-Val Split ──────────────────────────
    groups_train = groups.iloc[train_idx]

    val_fraction = VAL_SIZE / (1 - TEST_SIZE)

    gss_val = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=RANDOM_STATE
    )

    train_idx2, val_idx = next(
        gss_val.split(X_train, y_train, groups_train)
    )

    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    X_train = X_train.iloc[train_idx2]
    y_train = y_train.iloc[train_idx2]

    # ── Logging ─────────────────────────────────
    log.info(f"Split -> train:{len(X_train)}  val:{len(X_val)}  test:{len(X_test)}")
    log.info(f"Train distribution:\n{y_train.value_counts().sort_index()}")
    log.info(f"Val distribution:\n{y_val.value_counts().sort_index()}")
    log.info(f"Test distribution:\n{y_test.value_counts().sort_index()}")

    return X_train, X_val, X_test, y_train, y_val, y_test