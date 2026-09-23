"""Evaluation and report generation for Venjex Smart Lathi — Person B.

Generates all figures required by reports/REPORT.md:
  1. reports/figures/confusion_matrix.png
  2. reports/figures/feature_importance.png
  3. reports/figures/novelty_roc.png

Also prints the full sklearn classification report to stdout.

Usage
-----
    from src.evaluate import run_evaluation
    run_evaluation(pipeline, X_clean, y, feat_names, novelty_gate, X_outlier)
    
Or via CLI after training:
    python src/evaluate.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on Pi and headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = os.path.join("reports", "figures")

# Consistent colour palette for the 4 canonical classes
CLASS_COLOURS = {
    "clean":         "#4CAF50",   # green
    "benign_odour":  "#FFC107",   # amber
    "narcotic_voc":  "#F44336",   # red
    "nitrate_trace": "#9C27B0",   # purple
}


# ---------------------------------------------------------------------------
# 1. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_labels,
    save_path=None,
    title="Venjex Smart Lathi — Confusion Matrix",
):
    """Plot and save a normalised confusion matrix heatmap.

    Args:
        y_true:       Ground-truth label array.
        y_pred:       Predicted label array.
        class_labels: Ordered list of class names for axis ticks.
        save_path:    File path to save PNG. Defaults to figures dir.
        title:        Plot title string.

    Returns:
        str: Absolute path of saved figure.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {os.path.abspath(save_path)}")
    return save_path


# ---------------------------------------------------------------------------
# 2. Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    pipeline,
    feat_names,
    top_n=20,
    save_path=None,
    title="Top Feature Importances — Random Forest / Gradient Boosting",
):
    """Plot top-N feature importances from a tree-based pipeline step.

    Args:
        pipeline:   Fitted sklearn Pipeline containing a 'rf' or 'gb' step.
        feat_names: List of feature name strings (length == n_features).
        top_n:      Number of top features to display.
        save_path:  Destination PNG path.
        title:      Plot title.

    Returns:
        str | None: Absolute path of saved figure, or None if not applicable.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "feature_importance.png")

    # Retrieve the tree model step
    step_model = (
        pipeline.named_steps.get("rf")
        or pipeline.named_steps.get("gb")
    )
    if step_model is None or not hasattr(step_model, "feature_importances_"):
        print("[evaluate] No tree-based step found — skipping feature importance plot.")
        return None

    importances = step_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feat_names[i] for i in indices]
    top_vals = importances[indices]

    colours = ["#E53935" if v > np.percentile(top_vals, 75) else "#1E88E5" for v in top_vals]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    bars = ax.barh(range(top_n), top_vals[::-1], color=colours[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Feature importance plot saved → {os.path.abspath(save_path)}")
    return save_path


# ---------------------------------------------------------------------------
# 3. Novelty ROC Curve
# ---------------------------------------------------------------------------

def plot_novelty_roc(
    novelty_gate,
    X_inlier,
    X_outlier,
    save_path=None,
    title="Novelty Gate — ROC Curve (IsolationForest)",
):
    """Plot and save the ROC curve for the novelty/OOD detection gate.

    Args:
        novelty_gate: Fitted NoveltyGate instance.
        X_inlier:     In-distribution feature matrix (training scans).
        X_outlier:    Out-of-distribution feature matrix (synthetic noise).
        save_path:    Destination PNG path.
        title:        Plot title.

    Returns:
        str: Absolute path of saved figure.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "novelty_roc.png")

    fpr, tpr, _, auc = novelty_gate.compute_roc(X_inlier, X_outlier)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"IsolationForest (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E53935")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (Normal classified as Novel)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Novel correctly detected)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Novelty ROC plot saved → {os.path.abspath(save_path)}")
    return save_path


# ---------------------------------------------------------------------------
# 4. Classification Report (text)
# ---------------------------------------------------------------------------

def print_classification_report(y_true, y_pred, class_labels=None):
    """Print sklearn classification report with a header."""
    print("\n" + "=" * 65)
    print("  VENJEX SMART LATHI — FULL DATASET CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_true, y_pred, labels=class_labels, zero_division=0))


# ---------------------------------------------------------------------------
# 5. Full evaluation runner (called by src/train.py)
# ---------------------------------------------------------------------------

def run_evaluation(
    pipeline,
    X_clean,
    y,
    feat_names,
    novelty_gate,
    class_labels=None,
):
    """Run full Person B evaluation suite and save all report figures.

    Args:
        pipeline:     Fitted production sklearn Pipeline.
        X_clean:      Drift-corrected feature matrix used for training.
        y:            Label array corresponding to X_clean rows.
        feat_names:   List of 68 feature name strings.
        novelty_gate: Fitted NoveltyGate instance.
        class_labels: Ordered class list for confusion matrix axes.

    Returns:
        dict: Paths of all saved figures.
    """
    y = np.array(y)
    if class_labels is None:
        class_labels = sorted(set(y))

    # In-distribution predictions on the training set
    y_pred = pipeline.predict(X_clean)

    # Text report
    print_classification_report(y, y_pred, class_labels)

    # Generate a small synthetic OOD set for the ROC plot
    rng = np.random.default_rng(0)
    X_outlier_synth = rng.uniform(
        low=X_clean.min(axis=0) - 3 * X_clean.std(axis=0),
        high=X_clean.max(axis=0) + 3 * X_clean.std(axis=0),
        size=(max(50, len(X_clean) // 4), X_clean.shape[1]),
    )

    paths = {}
    paths["confusion_matrix"] = plot_confusion_matrix(y, y_pred, class_labels)
    paths["feature_importance"] = plot_feature_importance(pipeline, feat_names)
    paths["novelty_roc"] = plot_novelty_roc(novelty_gate, X_clean, X_outlier_synth)

    print("\n[evaluate] All report figures generated successfully.")
    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.bundle import load_bundle
    from src.features import FEATURE_NAMES, extract_batch, load_spec
    import pandas as pd

    print("Loading model bundle...")
    bundle = load_bundle("models/model_bundle.joblib")
    pipeline = bundle["pipeline"]
    novelty_gate = bundle["novelty_gate"]
    classes = bundle["classes"]
    feat_names = bundle["feature_names"]

    print("Loading processed dataset...")
    csv_path = os.path.join("data", "processed", "unified_dataset.csv")
    df = pd.read_csv(csv_path)
    spec = load_spec()
    X, y, _, _ = extract_batch(df, spec=spec)
    y = np.array(y)

    from src.drift import compensate_t_rh
    df_scans = df.groupby("scan_id", sort=False).first().loc[
        df.groupby("scan_id", sort=False).groups.keys()
    ]
    temps = df_scans["temp_c"].fillna(25.0).values[: len(y)]
    rhs = df_scans["rh_pct"].fillna(50.0).values[: len(y)]
    X_clean = compensate_t_rh(X, temps, rhs, coef=bundle["drift_coef"])

    run_evaluation(pipeline, X_clean, y, feat_names, novelty_gate, class_labels=classes)
