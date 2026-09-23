"""Novelty / Out-of-Distribution (OOD) detection gate for Venjex Smart Lathi.

Purpose
-------
Before classifying a scan as narcotic_voc / nitrate_trace / etc., we first
check if the feature vector actually looks like a real sensor scan. A faulty
sensor, a disconnected probe, or an unusual environment can produce feature
vectors that are so far outside the training distribution that the classifier's
output is meaningless.

The NoveltyGate uses an IsolationForest to learn the boundary of "normal"
sensor responses. If a new scan is flagged as anomalous, the downstream
classifier is NOT trusted and an OOD warning is raised instead.

Usage (training)
----------------
    from src.novelty import NoveltyGate
    gate = NoveltyGate()
    gate.fit(X_clean)           # X_clean: drift-corrected training features
    # gate is saved inside model_bundle.joblib by src/train.py

Usage (inference)
-----------------
    result = gate.predict_one(x_vec)
    # result = {"is_novel": False, "anomaly_score": 0.12}
    if result["is_novel"]:
        raise RuntimeError("OOD scan — do not trust classifier output")
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve


class NoveltyGate:
    """IsolationForest-based out-of-distribution detector.

    Attributes:
        contamination: Expected fraction of outliers in training set.
        random_state:  Reproducibility seed.
        threshold:     Decision threshold on anomaly score. Scores below this
                       are flagged as novel. Auto-set after fit() via
                       set_threshold_at_fpr().
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.threshold = None  # set by set_threshold_at_fpr()

        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=200,
                max_samples="auto",
            )),
        ])

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "NoveltyGate":
        """Fit the IsolationForest on drift-corrected training features.

        Args:
            X: Feature matrix of shape (N_scans, N_features).
               Should be the same X_clean passed to the classifier.

        Returns:
            self
        """
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self._pipe.fit(X_arr)
        # Default threshold: use IsolationForest's built-in contamination
        scores = self._pipe.decision_function(X_arr)
        self.threshold = float(np.percentile(scores, self.contamination * 100))
        print(
            f"[NoveltyGate] Fitted on {X_arr.shape[0]} scans. "
            f"Anomaly threshold (p{self.contamination*100:.0f}): {self.threshold:.4f}"
        )
        return self

    def set_threshold_at_fpr(
        self,
        X_inlier: np.ndarray,
        X_outlier: np.ndarray,
        target_fpr: float = 0.05,
    ) -> float:
        """Tune the decision threshold to achieve a target false-positive rate.

        Args:
            X_inlier:   Known in-distribution samples (training scans).
            X_outlier:  Known out-of-distribution samples (e.g. random noise).
            target_fpr: Desired FPR on inlier set (default 5%).

        Returns:
            Selected threshold value.
        """
        X_in = np.nan_to_num(np.asarray(X_inlier, dtype=float), nan=0.0)
        X_out = np.nan_to_num(np.asarray(X_outlier, dtype=float), nan=0.0)

        scores_in = self._pipe.decision_function(X_in)
        scores_out = self._pipe.decision_function(X_out)

        all_scores = np.concatenate([scores_in, scores_out])
        # IsolationForest: higher score = more normal. Inlier label = 1, outlier = 0
        labels = np.concatenate([np.ones(len(X_in)), np.zeros(len(X_out))])

        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        valid = fpr <= target_fpr
        if not np.any(valid):
            self.threshold = float(thresholds[0])
        else:
            self.threshold = float(thresholds[valid][-1])

        roc_auc = roc_auc_score(labels, all_scores)
        print(
            f"[NoveltyGate] Threshold tuned @ FPR={target_fpr:.2f}: "
            f"threshold={self.threshold:.4f}, ROC-AUC={roc_auc:.3f}"
        )
        return self.threshold

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def predict_one(self, x: np.ndarray) -> dict:
        """Run OOD check on a single feature vector.

        Args:
            x: 1-D feature array of length N_features, or (1, N_features).

        Returns:
            dict with keys:
                "is_novel"      : bool  — True if scan is OOD (do not trust classifier)
                "anomaly_score" : float — raw IsolationForest score (higher = more normal)
        """
        x_arr = np.nan_to_num(np.asarray(x, dtype=float).reshape(1, -1), nan=0.0)
        score = float(self._pipe.decision_function(x_arr)[0])
        is_novel = (score < self.threshold) if self.threshold is not None else (
            self._pipe.predict(x_arr)[0] == -1
        )
        return {
            "is_novel": bool(is_novel),
            "anomaly_score": score,
        }

    def predict_batch(self, X: np.ndarray) -> list[dict]:
        """Run OOD check on a batch of feature vectors.

        Args:
            X: Feature matrix of shape (N, F).

        Returns:
            List of dicts with "is_novel" and "anomaly_score" per sample.
        """
        X_arr = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        scores = self._pipe.decision_function(X_arr)
        results = []
        for score in scores:
            is_novel = (score < self.threshold) if self.threshold is not None else (score < 0)
            results.append({"is_novel": bool(is_novel), "anomaly_score": float(score)})
        return results

    # ------------------------------------------------------------------
    # Evaluation API (used by src/evaluate.py)
    # ------------------------------------------------------------------

    def compute_roc(
        self, X_inlier: np.ndarray, X_outlier: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute ROC curve for novelty detection.

        Args:
            X_inlier:  In-distribution feature matrix.
            X_outlier: Out-of-distribution feature matrix.

        Returns:
            (fpr, tpr, thresholds, roc_auc)
        """
        X_in = np.nan_to_num(np.asarray(X_inlier, dtype=float), nan=0.0)
        X_out = np.nan_to_num(np.asarray(X_outlier, dtype=float), nan=0.0)

        scores_in = self._pipe.decision_function(X_in)
        scores_out = self._pipe.decision_function(X_out)

        all_scores = np.concatenate([scores_in, scores_out])
        labels = np.concatenate([np.ones(len(X_in)), np.zeros(len(X_out))])

        fpr, tpr, thresholds = roc_curve(labels, all_scores)
        auc = roc_auc_score(labels, all_scores)
        return fpr, tpr, thresholds, auc
