"""Model bundle utilities for Venjex Smart Lathi.

The model_bundle.joblib is the SINGLE artifact shared between Person B (training)
and Person C (Pi inference). It contains everything Person C needs in one file.

Bundle Schema
-------------
{
    "pipeline":       sklearn Pipeline  — imputer + scaler + classifier
    "drift_coef":     np.ndarray (3, F) — T/RH least-squares coefficients
    "novelty_gate":   NoveltyGate       — IsolationForest OOD detector
    "feature_names":  list[str]         — 68 frozen feature names (THE CONTRACT)
    "classes":        list[str]         — canonical class label list
    "trained_at":     str               — ISO-8601 timestamp
    "model_name":     str               — winning model name (e.g. "Random Forest")
    "train_accuracy": float             — full-dataset accuracy after final fit
    "cv_macro_f1":    float             — mean Macro F1 from StratifiedKFold
    "n_train_scans":  int               — number of scans used in final training
    "version":        str               — bundle schema version
}
"""
import datetime
import os
import joblib
import numpy as np


BUNDLE_VERSION = "1.0.0"
_REQUIRED_KEYS = [
    "pipeline",
    "drift_coef",
    "novelty_gate",
    "feature_names",
    "classes",
]


def save_bundle(
    pipeline,
    drift_coef,
    novelty_gate,
    feature_names,
    classes,
    path="models/model_bundle.joblib",
    **meta,
):
    """Serialize the full model bundle to a single joblib file.

    Args:
        pipeline:      Trained sklearn Pipeline (imputer → [scaler] → classifier).
        drift_coef:    np.ndarray of shape (3, n_features) from compensate_t_rh().
        novelty_gate:  Fitted NoveltyGate instance from src.novelty.
        feature_names: Frozen list of 68 feature name strings.
        classes:       List of canonical class label strings.
        path:          Destination path for the .joblib file.
        **meta:        Optional metadata keys (model_name, cv_macro_f1, etc.)

    Returns:
        dict: The bundle dictionary that was saved.
    """
    bundle = {
        "pipeline":      pipeline,
        "drift_coef":    np.asarray(drift_coef, dtype=float),
        "novelty_gate":  novelty_gate,
        "feature_names": list(feature_names),
        "classes":       list(classes),
        "trained_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        "version":       BUNDLE_VERSION,
    }
    bundle.update(meta)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(bundle, path)
    print(f"[bundle] Saved model_bundle.joblib → {os.path.abspath(path)}")
    _print_summary(bundle)
    return bundle


def load_bundle(path="models/model_bundle.joblib"):
    """Load and validate a model bundle from disk.

    Args:
        path: Path to the .joblib file.

    Returns:
        dict: Bundle dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required keys are missing (incompatible bundle version).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'.\n"
            "Run 'python src/train.py' to generate it, or\n"
            "'python scripts/make_dummy_bundle.py' for a Day-1 placeholder."
        )

    bundle = joblib.load(path)

    missing = [k for k in _REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(
            f"Incompatible bundle (missing keys: {missing}). "
            f"Bundle version: {bundle.get('version', 'unknown')}. "
            "Please re-run training to regenerate."
        )

    print(f"[bundle] Loaded model_bundle.joblib from {os.path.abspath(path)}")
    _print_summary(bundle)
    return bundle


def _print_summary(bundle):
    """Print a concise human-readable bundle summary."""
    print(
        f"  version      : {bundle.get('version', '?')}\n"
        f"  trained_at   : {bundle.get('trained_at', '?')}\n"
        f"  model_name   : {bundle.get('model_name', '?')}\n"
        f"  classes      : {bundle.get('classes', [])}\n"
        f"  n_features   : {len(bundle.get('feature_names', []))}\n"
        f"  cv_macro_f1  : {bundle.get('cv_macro_f1', '?')}\n"
        f"  train_acc    : {bundle.get('train_accuracy', '?')}\n"
        f"  n_train_scans: {bundle.get('n_train_scans', '?')}"
    )
