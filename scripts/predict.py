"""Inference and threat assessment module for Venjex Smart Lathi.

Provides real-time single-scan classification and threat reporting:
- Extracts 68 contract features using src.features.
- Applies baseline & T/RH drift compensation from src.drift.
- Queries models/lathi_classifier.joblib for threat probability distribution.
- Can be used as a CLI tool or imported as a Python class (LathiPredictor).
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract, load_spec, FEATURE_NAMES
from src.drift import compensate_t_rh


THREAT_LEVELS = {
    "narcotic_voc": {
        "level": "CRITICAL THREAT",
        "description": "Narcotic Volatile Organic Compound Detected (Proxy: Ethanol/Acetone/Toluene)",
        "color": "[RED ALERT]",
    },
    "nitrate_trace": {
        "level": "CRITICAL THREAT",
        "description": "Explosive Precursor Trace Detected (Proxy: Ammonia/Urea/Nitrate)",
        "color": "[RED ALERT]",
    },
    "benign_odour": {
        "level": "BENIGN",
        "description": "Ambient Background / False Alarm Odour (Perfume/Food/Cleaner)",
        "color": "[YELLOW NOTICE]",
    },
    "clean": {
        "level": "SAFE",
        "description": "Clean Ambient Air (Baseline Level)",
        "color": "[GREEN SAFE]",
    },
}


class LathiPredictor:
    """Production predictor wrapper for Smart Lathi stick hardware scans."""

    def __init__(self, model_path=None, spec_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "lathi_classifier.joblib")
        if spec_path is None:
            spec_path = os.path.join(base_dir, "config", "sensors.yaml")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Please run 'python scripts/train_model.py' to generate the model artifact."
            )

        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.spec = load_spec(spec_path)
        self.classes_ = list(self.model.classes_)
        
        coef_path = os.path.join(os.path.dirname(model_path), "drift_coef.npy")
        self.drift_coef = np.load(coef_path) if os.path.exists(coef_path) else None

    def predict_scan(self, scan_df, scan_id=None):
        """Run end-to-end inference on a single scan DataFrame.
        
        Args:
            scan_df: DataFrame in unified long format for a single scan.
            scan_id: Optional identifier string.
            
        Returns:
            Dictionary containing prediction, confidence, probabilities, and threat assessment.
        """
        if scan_id is None:
            scan_id = scan_df["scan_id"].iloc[0] if "scan_id" in scan_df.columns else "unknown_scan"

        # 1. Feature Extraction (Contract 68 features)
        vec, feat_names = extract(scan_df, spec=self.spec)
        X = vec.reshape(1, -1)

        # 2. Environmental Readings & Drift Compensation
        temp = float(scan_df["temp_c"].median()) if "temp_c" in scan_df.columns and not scan_df["temp_c"].isna().all() else 25.0
        rh = float(scan_df["rh_pct"].median()) if "rh_pct" in scan_df.columns and not scan_df["rh_pct"].isna().all() else 50.0

        X_clean = compensate_t_rh(X, [temp], [rh], coef=self.drift_coef)

        # 3. Model Classification
        pred_class = self.model.predict(X_clean)[0]

        # 4. Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_clean)[0]
        else:
            probs = np.zeros(len(self.classes_))
            pred_idx = self.classes_.index(pred_class)
            probs[pred_idx] = 1.0

        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        confidence = float(prob_dict.get(pred_class, 1.0))

        # 5. Threat Assessment Mapping
        meta = THREAT_LEVELS.get(pred_class, {
            "level": "UNKNOWN",
            "description": "Unclassified chemical pattern",
            "color": "[GREY]",
        })

        return {
            "scan_id": str(scan_id),
            "predicted_class": pred_class,
            "confidence": confidence,
            "threat_level": meta["level"],
            "description": meta["description"],
            "status_tag": meta["color"],
            "probabilities": prob_dict,
            "environmental": {
                "temp_c": temp,
                "rh_pct": rh,
            },
        }

    def format_report(self, result):
        """Format prediction dictionary into human-readable terminal report."""
        lines = [
            "=" * 65,
            f"  VENJEX SMART LATHI — CHEMICAL SCAN REPORT",
            "=" * 65,
            f"  Scan Identifier   : {result['scan_id']}",
            f"  Detection Status  : {result['status_tag']} {result['threat_level']}",
            f"  Identified Class  : {result['predicted_class'].upper()}",
            f"  Confidence Score  : {result['confidence'] * 100:.1f}%",
            f"  Detail            : {result['description']}",
            f"  Ambient Temp / RH : {result['environmental']['temp_c']:.1f}°C / {result['environmental']['rh_pct']:.1f}%",
            "-" * 65,
            "  Class Probability Breakdown:",
        ]
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob * 30)
            lines.append(f"    - {cls:<16} : {prob * 100:5.1f}%  |{bar:<30}|")
        lines.append("=" * 65)
        return "\n".join(lines)


def run_demo():
    """Run interactive inference demo on diverse scans from the processed dataset."""
    print("Initializing Venjex Smart Lathi Predictor...")
    predictor = LathiPredictor()

    csv_path = os.path.join("data", "processed", "unified_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset {csv_path} not found. Please run 'python scripts/verify_pipeline.py' first.")
        return

    df = pd.read_csv(csv_path)
    
    # Select demonstrative scans (including transient MOX curves)
    demo_scans = []
    for target_id in ["transient_scan_2", "transient_scan_1", "bme_coffee_s1_matrix", "uci_b1_s1"]:
        sub = df[df["scan_id"] == target_id]
        if len(sub) > 0:
            demo_scans.append((target_id, sub))

    # Fallback to any canonical class if specific IDs not found
    if len(demo_scans) < 2:
        for scan_id, grp in df.groupby("scan_id", sort=False):
            demo_scans.append((scan_id, grp))
            if len(demo_scans) == 4:
                break

    print(f"\nRunning test inferences on {len(demo_scans)} representative sample scans:\n")
    for scan_id, scan_df in demo_scans:
        res = predictor.predict_scan(scan_df, scan_id=scan_id)
        print(predictor.format_report(res))
        print()


def main():
    parser = argparse.ArgumentParser(description="Venjex Smart Lathi — Real-Time Inference CLI")
    parser.add_argument("--file", type=str, help="Path to raw scan CSV file")
    parser.add_argument("--scan-id", type=str, help="Specific scan_id from processed dataset to test")
    args = parser.parse_args()

    predictor = LathiPredictor()

    if args.file:
        df_scan = pd.read_csv(args.file)
        result = predictor.predict_scan(df_scan)
        print(predictor.format_report(result))
    elif args.scan_id:
        csv_path = os.path.join("data", "processed", "unified_dataset.csv")
        df = pd.read_csv(csv_path)
        sub = df[df["scan_id"] == args.scan_id]
        if len(sub) == 0:
            print(f"Scan ID '{args.scan_id}' not found in {csv_path}!")
            return
        result = predictor.predict_scan(sub, scan_id=args.scan_id)
        print(predictor.format_report(result))
    else:
        run_demo()


if __name__ == "__main__":
    main()
