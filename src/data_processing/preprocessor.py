import os, sys, logging
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_PROCESSED

log = logging.getLogger(__name__)


CAT_ENCODE_COLS = [
    "Domain",                  
    "Mutational_event",         
    "Variant_Classification",   
    "Variant_Type",            
    "Sample_origin",          
    "Complexity",           
    "Genome_base_coding",      
    "WT_AA_1",                
    "Mutant_AA_1",             
    "WT_AA_3",                  
    "Mutant_AA_3",             
    "Mutant_Allele",          
    "Wt_Codon",                
    "Mutant_Codon",             
    "Internal",                
    "Solid",                   
]


DROP_INSIDE = [
    "Disease",                  
    "Variant Comment",          
    "Mutassessor_prediction:",  
    "HG38_Variant",             
    "NG_017013.2_Variant",      
]


class P53Preprocessor:
    def __init__(self):
        self.pipeline      = None
        self.feature_names = []
        self.num_cols_     = []
        self.cat_cols_     = []

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X.drop(columns=[c for c in DROP_INSIDE if c in X.columns],
               inplace=True, errors="ignore")
        return X

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._prepare(X)

        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [c for c in CAT_ENCODE_COLS if c in X.columns]

        log.info(f"Numeric: {len(self.num_cols_)}  Categorical: {len(self.cat_cols_)}")

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False,
                                      max_categories=30)),
        ])

        transformers = [("num", num_pipe, self.num_cols_)]
        if self.cat_cols_:
            transformers.append(("cat", cat_pipe, self.cat_cols_))

        self.pipeline = ColumnTransformer(transformers, remainder="drop")
        X_proc = self.pipeline.fit_transform(X)

        self.feature_names = list(self.num_cols_)
        if self.cat_cols_:
            ohe = self.pipeline.named_transformers_["cat"]["ohe"]
            self.feature_names += ohe.get_feature_names_out(
                self.cat_cols_).tolist()

        log.info(f"Preprocessed shape: {X_proc.shape}")
        return X_proc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Call fit_transform() before transform().")
        X = self._prepare(X)
        return self.pipeline.transform(X)

    @staticmethod
    def compute_class_weights(y: np.ndarray) -> dict:
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y)
        weight_dict = dict(zip(classes.tolist(), weights.tolist()))
        log.info(f"Class weights: {weight_dict}")
        return weight_dict

    def save(self, path: str = None):
        path = path or os.path.join(DATA_PROCESSED, "preprocessor.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Preprocessor saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = path or os.path.join(DATA_PROCESSED, "preprocessor.pkl")
        return joblib.load(path)
