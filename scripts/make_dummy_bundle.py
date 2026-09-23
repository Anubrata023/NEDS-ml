"""Make Dummy Bundle — Day-1 placeholder for Person C.

Generates a minimal model_bundle.joblib that Person C can use immediately
to build and test pi/inference.py — before Person B has real training data.

The dummy bundle:
  - Always predicts "clean" (safe default)
  - Returns 25% probability for each class (uniform)
  - Has a dummy NoveltyGate that never flags novel scans
  - Has zero drift coefficients (no correction applied)
  - Is clearly marked as dummy in metadata

Usage
-----
    python scripts/make_dummy_bundle.py
    python scripts/make_dummy_bundle.py --out models/model_bundle.joblib
"""
import os
import sys
import argparse
import numpy as np
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_NAMES, load_spec
from src.novelty import NoveltyGate
from src.bundle import save_bundle, BUNDLE_VERSION


# ---------------------------------------------------------------------------
# Dummy sklearn-compatible pipeline
# ---------------------------------------------------------------------------

class _DummyClassifier:
    """Always predicts 'clean' with uniform probability — never trains."""

    def __init__(self, classes):
        self.classes_ = classes

    def predict(self, X):
        return np.array(["clean"] * len(X))

    def predict_proba(self, X):
        n = len(self.classes_)
        return np.full((len(X), n), 1.0 / n)


class _DummyPipeline:
    """Wraps DummyClassifier to behave like a sklearn Pipeline."""

    def __init__(self, classes):
        self._clf = _DummyClassifier(classes)
        self.classes_ = classes
        self.named_steps = {}  # no tree steps → evaluate.py skips feature importance

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ---------------------------------------------------------------------------
# Dummy novelty gate
# ---------------------------------------------------------------------------

class _DummyNoveltyGate:
    """Never flags any scan as novel — always returns in-distribution."""

    threshold = 0.0

    def predict_one(self, x):
        return {"is_novel": False, "anomaly_score": 0.5}

    def predict_batch(self, X):
        return [{"is_novel": False, "anomaly_score": 0.5} for _ in range(len(X))]

    def compute_roc(self, X_inlier, X_outlier):
        # Returns a trivial ROC: diagonal (random classifier)
        fpr = np.linspace(0, 1, 50)
        tpr = fpr.copy()
        thresholds = np.linspace(1, 0, 50)
        return fpr, tpr, thresholds, 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CANONICAL_CLASSES = ["benign_odour", "clean", "narcotic_voc", "nitrate_trace"]
N_FEATURES = len(FEATURE_NAMES)


def make_dummy(out_path="models/model_bundle.joblib"):
    """Create and save the Day-1 dummy model bundle.

    Args:
        out_path: Destination path for the .joblib file.
    """
    print("=" * 60)
    print("  VENJEX SMART LATHI — MAKING DUMMY BUNDLE (Day-1 Stub)")
    print("=" * 60)

    pipeline     = _DummyPipeline(CANONICAL_CLASSES)
    drift_coef   = np.zeros((3, N_FEATURES), dtype=float)
    novelty_gate = _DummyNoveltyGate()

    bundle = save_bundle(
        pipeline=pipeline,
        drift_coef=drift_coef,
        novelty_gate=novelty_gate,
        feature_names=FEATURE_NAMES,
        classes=CANONICAL_CLASSES,
        path=out_path,
        model_name="DUMMY (always predicts clean)",
        cv_macro_f1=0.0,
        train_accuracy=0.0,
        n_train_scans=0,
        is_dummy=True,
        dummy_note=(
            "This is a Day-1 placeholder bundle for Person C to use while "
            "Person B trains the real model. Replace with: python src/train.py"
        ),
    )

    print("\n  ✓ Dummy bundle created successfully.")
    print(f"  Person C: load with  from src.bundle import load_bundle")
    print(f"  Replace with real:   python src/train.py")
    print("=" * 60)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Day-1 dummy model_bundle.joblib for Person C"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/model_bundle.joblib",
        help="Output path for the dummy bundle",
    )
    args = parser.parse_args()
    make_dummy(out_path=args.out)
