import os, sys, logging
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from config import XGB_PARAMS, RF_PARAMS, MODEL_DIR as MODELS_DIR, NUM_CLASSES

log = logging.getLogger(__name__)


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:

    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y)
    weight_map = dict(zip(classes.tolist(), class_weights.tolist()))
    return np.array([weight_map[label] for label in y])


class XGBoostModel:
    def __init__(self, params: dict = None):
        self.model = XGBClassifier(**(params or XGB_PARAMS))

    def fit(self, X_train, y_train, X_val=None, y_val=None):

        sample_weights = _compute_sample_weights(y_train)
        log.info("XGBoost training with class-balanced sample weights")

        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,   
            eval_set=eval_set,
            verbose=False,
        )
        log.info("XGBoost training complete")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)        

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        proba = self.predict_proba(X)
        preds = self.predict(X)
        auc  = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        f1   = f1_score(y, preds, average="weighted")
        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, average="weighted", zero_division=0)
        rec  = recall_score(y, preds, average="weighted", zero_division=0)
        log.info(f"XGBoost -> AUC:{auc:.4f}  F1:{f1:.4f}  "
                 f"Acc:{acc:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}")
        return {"auc": auc, "f1": f1, "accuracy": acc,
                "precision": prec, "recall": rec}

    def feature_importance(self, feature_names: list = None) -> dict:
        imp = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: x[1], reverse=True))

    def save(self, path: str = None):
        path = path or os.path.join(MODELS_DIR, "xgboost_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info(f"XGBoost saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        return joblib.load(path or os.path.join(MODELS_DIR, "xgboost_model.pkl"))


class RandomForestModel:
    def __init__(self, params: dict = None):

        rf_params = dict(params or RF_PARAMS)
        rf_params.setdefault("class_weight", "balanced")   
        self.model = RandomForestClassifier(**rf_params)

    def fit(self, X_train, y_train):

        sample_weights = _compute_sample_weights(y_train)
        log.info("Random Forest training with class-balanced sample weights")
        self.model.fit(X_train, y_train, sample_weight=sample_weights)  
        log.info("Random Forest training complete")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)       
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        proba = self.predict_proba(X)
        preds = self.predict(X)
        auc  = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        f1   = f1_score(y, preds, average="weighted")
        acc  = accuracy_score(y, preds)
        prec = precision_score(y, preds, average="weighted", zero_division=0)
        rec  = recall_score(y, preds, average="weighted", zero_division=0)
        log.info(f"Random Forest -> AUC:{auc:.4f}  F1:{f1:.4f}  "
                 f"Acc:{acc:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}")
        return {"auc": auc, "f1": f1, "accuracy": acc,
                "precision": prec, "recall": rec}

    def feature_importance(self, feature_names: list = None) -> dict:
        imp = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: x[1], reverse=True))

    def save(self, path: str = None):
        path = path or os.path.join(MODELS_DIR, "rf_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Random Forest saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        return joblib.load(path or os.path.join(MODELS_DIR, "rf_model.pkl"))
