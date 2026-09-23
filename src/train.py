"""Stage-B Model Bake-off, Training, and Bundle Serialization — Person B.

Workflow
--------
1. Loads processed dataset (Parquet preferred, CSV fallback).
2. Extracts 68-feature frozen contract via src.features.
3. Applies T/RH drift compensation via src.drift.
4. Runs bake-off across RF / GBT / SVM with StratifiedKFold + GroupKFold.
5. Retrains the best model on the full dataset.
6. Fits a NoveltyGate (IsolationForest) on the full clean feature set.
7. Runs full evaluation → saves confusion matrix, feature importance, novelty ROC.
8. Saves complete model_bundle.joblib via src.bundle.

Usage
-----
    python src/train.py
    python src/train.py --data data/processed/unified_dataset.parquet
    python src/train.py --out models/model_bundle.joblib
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Workspace root on sys.path so `src.*` imports work from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_batch, load_spec, FEATURE_NAMES
from src.drift import compensate_t_rh
from src.novelty import NoveltyGate
from src.bundle import save_bundle
from src.evaluate import run_evaluation


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_path=None):
    """Load unified dataset, preferring Parquet over CSV.

    Args:
        data_path: Optional explicit path. If None, searches default locations.

    Returns:
        pd.DataFrame: Unified long-format dataset.
    """
    candidates = []
    if data_path:
        candidates.append(data_path)
    candidates += [
        os.path.join("data", "processed", "unified_dataset.parquet"),
        os.path.join("data", "ours", "stage_b_dataset.parquet"),
        os.path.join("data", "processed", "unified_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[train] Loading dataset: {path}")
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)

    raise FileNotFoundError(
        "No processed dataset found!\n"
        "Run 'python scripts/verify_pipeline.py' to generate a synthetic one, or\n"
        "place your Stage-B data at data/ours/stage_b_dataset.parquet."
    )


# ---------------------------------------------------------------------------
# Model candidates
# ---------------------------------------------------------------------------

def _build_candidates():
    """Return dict of named sklearn Pipeline candidates for the bake-off."""
    return {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("gb", GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.85,
                random_state=42,
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)),
        ]),
    }


# ---------------------------------------------------------------------------
# Bake-off: StratifiedKFold
# ---------------------------------------------------------------------------

def stratified_bakeoff(X, y, n_splits=5):
    """Run StratifiedKFold bake-off across all model candidates.

    Args:
        X:        Drift-corrected feature matrix.
        y:        Label array.
        n_splits: Number of CV folds.

    Returns:
        tuple: (best_model_name, best_pipeline, cv_results_dict)
    """
    print("\n" + "─" * 65)
    print(f"  BAKE-OFF: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print("─" * 65)

    candidates = _build_candidates()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    best_f1 = -1.0
    best_name = None
    best_pipe = None

    for name, pipe in candidates.items():
        acc_scores, f1_scores = [], []
        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[test_idx])
            acc_scores.append(accuracy_score(y[test_idx], preds))
            f1_scores.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))

        mean_acc = float(np.mean(acc_scores))
        mean_f1  = float(np.mean(f1_scores))
        std_f1   = float(np.std(f1_scores))
        print(
            f"  [{name:20s}]  Acc: {mean_acc*100:5.1f}%  "
            f"Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}"
        )
        results[name] = {"acc": mean_acc, "macro_f1": mean_f1, "std_f1": std_f1}

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_name = name
            best_pipe = pipe

    print(f"\n  ✓ Best model: [{best_name}]  Macro-F1 = {best_f1:.4f}")
    return best_name, best_pipe, results


# ---------------------------------------------------------------------------
# GroupKFold validation
# ---------------------------------------------------------------------------

def group_bakeoff(X, y, groups, group_label="collection_day"):
    """Run GroupKFold bake-off to test session/day generalisation.

    Args:
        X:           Drift-corrected feature matrix.
        y:           Label array.
        groups:      Group assignment array (e.g. collection_day integers).
        group_label: Human-readable name of grouping variable.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        print(f"\n[train] GroupKFold skipped — only 1 unique {group_label} in dataset.")
        print("        Collect data across multiple days for a valid GroupKFold result.")
        return

    n_splits = min(3, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    candidates = _build_candidates()

    print("\n" + "─" * 65)
    print(f"  GROUP-K-FOLD (grouped by {group_label}, n_splits={n_splits})")
    print("─" * 65)

    for name, pipe in candidates.items():
        scores = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            train_cls = set(y[train_idx])
            test_cls  = set(y[test_idx])
            if test_cls - train_cls:
                continue  # skip splits with unseen classes in test
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[test_idx])
            scores.append(accuracy_score(y[test_idx], preds))

        if scores:
            print(f"  [{name:20s}]  GroupKFold Acc: {np.mean(scores)*100:.1f}%")
        else:
            print(f"  [{name:20s}]  GroupKFold: disjoint classes (need more Stage-B data)")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(data_path=None, out_path="models/model_bundle.joblib"):
    """Full Person B training pipeline.

    Args:
        data_path: Optional path to dataset file.
        out_path:  Destination path for the model_bundle.joblib.
    """
    print("=" * 65)
    print("  VENJEX SMART LATHI — PERSON B MODEL TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load data
    df = load_dataset(data_path)
    print(f"[train] Loaded {len(df):,} rows, {df['scan_id'].nunique():,} unique scans.")

    # 2. Feature extraction
    spec = load_spec()
    print(f"\n[train] Extracting {len(FEATURE_NAMES)}-feature contract...")
    X, y, scan_ids, feat_names = extract_batch(df, spec=spec)
    y = np.array(y)
    print(f"[train] Feature matrix: {X.shape}  |  Classes: {sorted(set(y))}")

    # 3. Align metadata for drift + grouping
    df_scans = df.groupby("scan_id", sort=False).first().loc[scan_ids]
    temps = df_scans["temp_c"].fillna(25.0).values
    rhs   = df_scans["rh_pct"].fillna(50.0).values
    days  = df_scans["collection_day"].fillna(1).astype(int).values
    rooms = df_scans["room_id"].fillna("unknown").astype(str).values

    # 4. Drift compensation
    print("\n[train] Applying T/RH drift compensation...")
    X_clean, drift_coef = compensate_t_rh(X, temps, rhs, return_coef=True)

    # 5. Bake-off
    best_name, best_pipe, cv_results = stratified_bakeoff(X_clean, y, n_splits=5)

    # 6. GroupKFold (by day, then by room)
    group_bakeoff(X_clean, y, days, "collection_day")
    group_bakeoff(X_clean, y, rooms, "room_id")

    # 7. Final production training on full dataset
    print("\n" + "─" * 65)
    print(f"  FINAL TRAINING: {best_name} on full dataset ({len(y)} scans)")
    print("─" * 65)
    best_pipe.fit(X_clean, y)
    train_preds = best_pipe.predict(X_clean)
    train_acc = float(accuracy_score(y, train_preds))
    print(f"  Full-dataset accuracy after final fit: {train_acc*100:.2f}%")
    print(classification_report(y, train_preds, zero_division=0))

    # 8. Novelty gate
    print("[train] Fitting IsolationForest novelty gate...")
    gate = NoveltyGate(contamination=0.05, random_state=42)
    gate.fit(X_clean)

    # 9. Evaluation figures
    print("\n[train] Generating evaluation figures...")
    run_evaluation(
        pipeline=best_pipe,
        X_clean=X_clean,
        y=y,
        feat_names=feat_names,
        novelty_gate=gate,
        class_labels=sorted(set(y)),
    )

    # 10. Save bundle
    bundle = save_bundle(
        pipeline=best_pipe,
        drift_coef=drift_coef,
        novelty_gate=gate,
        feature_names=feat_names,
        classes=sorted(set(y)),
        path=out_path,
        model_name=best_name,
        cv_macro_f1=cv_results[best_name]["macro_f1"],
        train_accuracy=train_acc,
        n_train_scans=int(len(y)),
    )

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE ✓")
    print(f"  Model bundle → {os.path.abspath(out_path)}")
    print("=" * 65)
    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Venjex Smart Lathi — Person B training pipeline"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to dataset file")
    parser.add_argument(
        "--out",
        type=str,
        default="models/model_bundle.joblib",
        help="Output path for model_bundle.joblib",
    )
    args = parser.parse_args()
    train(data_path=args.data, out_path=args.out)
