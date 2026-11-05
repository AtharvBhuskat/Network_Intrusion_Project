"""
train.py
Train anomaly detector (Isolation Forest) and family classifier (Logistic Regression).
Saves preprocessor.pkl, anomaly_model.pkl, family_classifier.pkl to models/.
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_processed_data():
    """Load train/test CSVs."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    print(f"[train] Loaded train: {train.shape}, test: {test.shape}")
    return train, test


def get_feature_cols(df):
    """Return numeric and categorical feature column lists."""
    exclude = {"label", "attack_family", "is_suspicious"}
    all_cols = [c for c in df.columns if c not in exclude]
    numeric = [c for c in all_cols if c not in CATEGORICAL_COLS]
    return numeric, CATEGORICAL_COLS, all_cols


def build_preprocessor(numeric_cols, categorical_cols):
    """Column transformer: RobustScaler for numeric, OneHotEncoder for categorical."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_anomaly_model(X_train_processed, y_suspicious_train):
    """Train Isolation Forest for anomaly (suspicion) detection."""
    contamination = y_suspicious_train.mean()
    contamination = min(max(contamination, 0.01), 0.45)

    iso = IsolationForest(
        n_estimators=200,
        max_samples=min(10000, len(X_train_processed)),
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_processed)
    print(f"[train] Isolation Forest trained (contamination={contamination:.3f})")
    return iso


def train_family_classifier(X_train_processed, y_family_train):
    """Train Logistic Regression for 5-class attack family prediction."""
    lr = LogisticRegression(
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train_processed, y_family_train)
    print(f"[train] Logistic Regression trained (classes: {list(lr.classes_)})")
    return lr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly(model, X_test, y_true):
    """Evaluate Isolation Forest."""
    raw_scores = model.decision_function(X_test)
    preds = model.predict(X_test)
    # IF returns 1=normal, -1=anomaly; convert to 0/1 suspicious
    pred_suspicious = (preds == -1).astype(int)

    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_true, pred_suspicious, zero_division=0)
    rec = recall_score(y_true, pred_suspicious, zero_division=0)

    try:
        auc = roc_auc_score(y_true, -raw_scores)
    except ValueError:
        auc = 0.0

    print("\n--- Anomaly Detection (Isolation Forest) ---")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    return {"precision": prec, "recall": rec, "roc_auc": auc,
            "predictions": pred_suspicious, "scores": raw_scores}


def evaluate_classifier(model, X_test, y_true, class_names):
    """Evaluate Logistic Regression classifier."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    report = classification_report(y_true, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, preds, labels=class_names)
    macro_f1 = f1_score(y_true, preds, average="macro")

    print("\n--- Family Classifier (Logistic Regression) ---")
    print(classification_report(y_true, preds, target_names=class_names))
    print(f"  Macro F1  : {macro_f1:.4f}")

    return {"report": report, "confusion_matrix": cm,
            "predictions": preds, "probabilities": proba,
            "macro_f1": macro_f1, "class_names": class_names}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_models(preprocessor, anomaly_model, classifier, eval_results):
    """Save all model artefacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(anomaly_model, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    joblib.dump(classifier, os.path.join(MODELS_DIR, "family_classifier.pkl"))
    joblib.dump(eval_results, os.path.join(MODELS_DIR, "eval_results.pkl"))

    print(f"\n[train] All models saved to {MODELS_DIR}/")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_training():
    """Full training pipeline."""
    # 1. Load data
    train_df, test_df = load_processed_data()
    numeric_cols, cat_cols, feature_cols = get_feature_cols(train_df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_suspicious_train = train_df["is_suspicious"]
    y_suspicious_test = test_df["is_suspicious"]
    y_family_train = train_df["attack_family"]
    y_family_test = test_df["attack_family"]

    # 2. Build and fit preprocessor
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    print("[train] Fitting preprocessor ...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"[train] Processed feature shape: {X_train_proc.shape}")

    # 3. Train models
    anomaly_model = train_anomaly_model(X_train_proc, y_suspicious_train)
    classifier = train_family_classifier(X_train_proc, y_family_train)

    # 4. Evaluate
    class_names = sorted(y_family_train.unique().tolist())
    anomaly_eval = evaluate_anomaly(anomaly_model, X_test_proc, y_suspicious_test)
    classifier_eval = evaluate_classifier(classifier, X_test_proc, y_family_test, class_names)

    eval_results = {
        "anomaly": anomaly_eval,
        "classifier": classifier_eval,
    }

    # 5. Save
    save_models(preprocessor, anomaly_model, classifier, eval_results)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Anomaly ROC-AUC : {anomaly_eval['roc_auc']:.4f}")
    print(f"Classifier F1   : {classifier_eval['macro_f1']:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    run_training()
