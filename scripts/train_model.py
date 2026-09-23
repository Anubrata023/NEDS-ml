"""Stage-B Model Training and Cross-Validation Pipeline — Venjex Smart Lathi.

Implements Person B workflow:
1. Loads processed dataset (Parquet / CSV).
2. Extracts 68 frozen contract features from src.features.
3. Applies T/RH drift compensation from src.drift.
4. Validates models using StratifiedKFold and GroupKFold (collection_day / room_id).
5. Reports Classification Report, Confusion Matrix, and Top 15 Feature Importances.
6. Serializes trained production artifact to models/lathi_classifier.joblib.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_batch, load_spec, FEATURE_NAMES
from src.drift import compensate_t_rh


def load_dataset():
    """Load unified dataset, preferring Parquet over CSV."""
    parquet_path = os.path.join("data", "processed", "unified_dataset.parquet")
    csv_path = os.path.join("data", "processed", "unified_dataset.csv")

    if os.path.exists(parquet_path):
        print(f"Loading dataset from {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Processed dataset not found! Please run 'python scripts/verify_pipeline.py' first."
        )


def train_and_evaluate():
    print("=" * 70)
    print("VENJEX SMART LATHI — MODEL TRAINING & VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = load_dataset()
    print(f"Total raw rows loaded: {len(df):,}")

    # 2. Extract Features
    spec = load_spec()
    print(f"\nExtracting features using frozen {len(FEATURE_NAMES)}-feature contract...")
    X, y, scan_ids, feat_names = extract_batch(df, spec=spec)
    y = np.array(y)
    print(f"Feature matrix shape: {X.shape} (Scans x Features)")

    # 3. Align Metadata for Drift Compensation & Grouping
    df_scans = df.groupby("scan_id", sort=False).first().loc[scan_ids]
    temps = df_scans["temp_c"].fillna(25.0).values
    rhs = df_scans["rh_pct"].fillna(50.0).values
    days = df_scans["collection_day"].fillna(1).astype(int).values
    rooms = df_scans["room_id"].fillna("unknown").astype(str).values

    # 4. Drift & Environmental Compensation
    print("\nApplying least-squares Temperature & Humidity baseline compensation...")
    X_clean, drift_coef = compensate_t_rh(X, temps, rhs, return_coef=True)

    # 5. Define Model Candidates
    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("gb", GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42)),
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=5.0, random_state=42)),
        ]),
    }

    # 6. Stratified Cross-Validation
    print("\n" + "-" * 70)
    print("1. STRATIFIED K-FOLD EVALUATION (Balanced Distribution Check)")
    print("-" * 70)

    best_score = -1.0
    best_model_name = None
    best_pipeline = None

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, pipe in models.items():
        acc_scores = []
        f1_scores = []
        all_preds = []
        all_true = []

        for train_idx, test_idx in skf.split(X_clean, y):
            pipe.fit(X_clean[train_idx], y[train_idx])
            preds = pipe.predict(X_clean[test_idx])
            acc_scores.append(accuracy_score(y[test_idx], preds))
            f1_scores.append(f1_score(y[test_idx], preds, average="macro"))
            all_preds.extend(preds)
            all_true.extend(y[test_idx])

        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        print(f"[{name}] Stratified Accuracy: {mean_acc * 100:.1f}% | Macro F1: {mean_f1:.3f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_model_name = name
            best_pipeline = pipe

    # 7. GroupKFold Cross-Validation (Collection Protocol Mandate)
    print("\n" + "-" * 70)
    print("2. GROUP-K-FOLD EVALUATION (Grouped by collection_day)")
    print("-" * 70)

    unique_days = np.unique(days)
    if len(unique_days) >= 2:
        n_splits = min(3, len(unique_days))
        gkf = GroupKFold(n_splits=n_splits)
        for name, pipe in models.items():
            g_scores = []
            for train_idx, test_idx in gkf.split(X_clean, y, groups=days):
                # Only evaluate if test set contains classes seen in train
                train_classes = set(y[train_idx])
                test_classes = set(y[test_idx])
                if len(test_classes - train_classes) == 0:
                    pipe.fit(X_clean[train_idx], y[train_idx])
                    preds = pipe.predict(X_clean[test_idx])
                    g_scores.append(accuracy_score(y[test_idx], preds))
            if g_scores:
                print(f"[{name}] GroupKFold (Days) Accuracy: {np.mean(g_scores) * 100:.1f}%")
            else:
                print(f"[{name}] GroupKFold (Days): Split contains disjoint classes (need Stage-B dataset)")
    else:
        print("Single day data detected. GroupKFold requires multi-day Stage-B collection.")

    # 8. Train Final Production Model & Feature Importance
    print("\n" + "-" * 70)
    print(f"3. FINAL MODEL TRAINING ({best_model_name}) & FEATURE IMPORTANCE")
    print("-" * 70)

    best_pipeline.fit(X_clean, y)

    # Detailed Classification Report on full set
    train_preds = best_pipeline.predict(X_clean)
    print("\nFull Dataset Classification Report:")
    print(classification_report(y, train_preds))

    # Feature Importances (Random Forest or Gradient Boosting)
    step_model = best_pipeline.named_steps.get("rf") or best_pipeline.named_steps.get("gb")
    if step_model is not None and hasattr(step_model, "feature_importances_"):
        importances = step_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        print("Top 15 Most Important Features for Classification:")
        for rank, idx in enumerate(sorted_indices[:15], 1):
            print(f"  {rank:2d}. {feat_names[idx]:<25} (Importance: {importances[idx]:.4f})")

    # 9. Save Model Artifact & Drift Coefficients
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "lathi_classifier.joblib")
    coef_save_path = os.path.join(models_dir, "drift_coef.npy")
    joblib.dump(best_pipeline, model_save_path)
    np.save(coef_save_path, drift_coef)
    print(f"\nTrained model successfully serialized to: {model_save_path}")
    print(f"Drift coefficients saved to: {coef_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
