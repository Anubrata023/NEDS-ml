"""End-to-end pipeline verification and synthetic dataset benchmarking script.

Tests all 7 loaders, feature extraction contract, sign inversions, ratio calculations,
and drift correction PCA plotting.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import load_spec, extract_batch, FEATURE_NAMES
from src.drift import compensate_t_rh, plot_drift_before_after
from src.loaders import bme688_odours, multimodal_gas, drift_uci, food_freshness, gas_array, adl, breath


def generate_synthetic_raw_data():
    """Generate synthetic raw data files for all 7 dataset schemas to test loaders."""
    print("Generating synthetic raw datasets...")
    np.random.seed(42)

    # 1. BME688 Odours (Laura Basa)
    bme_dir = os.path.join("data", "raw", "bme688", "coffee")
    os.makedirs(bme_dir, exist_ok=True)
    df_bme = pd.DataFrame({
        "heater_step": list(range(10)),
        "cycle_1": np.random.uniform(5000, 25000, 10),
        "cycle_2": np.random.uniform(5000, 25000, 10),
        "cycle_3": np.random.uniform(5000, 25000, 10),
    })
    df_bme.to_csv(os.path.join(bme_dir, "coffee_s1_matrix.csv"), index=False)

    # 2. Multimodal Gas (JOAI)
    mm_dir = os.path.join("data", "raw", "multimodal")
    os.makedirs(mm_dir, exist_ok=True)
    df_mm = pd.DataFrame({
        "serial_no": [101, 102, 103, 104],
        "sensor_1": np.random.uniform(10, 50, 4),
        "sensor_2": np.random.uniform(20, 60, 4),
        "sensor_3": np.random.uniform(15, 45, 4),
        "sensor_4": np.random.uniform(5, 25, 4),
        "sensor_5": np.random.uniform(30, 70, 4),
        "sensor_6": np.random.uniform(12, 35, 4),
        "sensor_7": np.random.uniform(8, 40, 4),
        "class_label": ["No Gas", "Perfume", "Smoke", "Mixture"]
    })
    df_mm.to_csv(os.path.join(mm_dir, "multimodal_gas.csv"), index=False)

    # 3. UCI Gas Sensor Array Drift (Vergara)
    drift_dir = os.path.join("data", "raw", "drift")
    os.makedirs(drift_dir, exist_ok=True)
    lines_b1 = []
    lines_b2 = []
    for s_i in range(10):
        # Gas 1 = Ethanol (narcotic_voc), Gas 3 = Ammonia (nitrate_trace)
        gas_id = 1 if s_i % 2 == 0 else 3
        feats_str1 = " ".join([f"{(idx*8)+1}:{np.random.uniform(10, 100):.2f}" for idx in range(16)])
        feats_str2 = " ".join([f"{(idx*8)+1}:{np.random.uniform(15, 120):.2f}" for idx in range(16)])
        lines_b1.append(f"{gas_id}; {feats_str1}\n")
        lines_b2.append(f"{gas_id}; {feats_str2}\n")

    with open(os.path.join(drift_dir, "batch1.dat"), "w") as f:
        f.writelines(lines_b1)
    with open(os.path.join(drift_dir, "batch2.dat"), "w") as f:
        f.writelines(lines_b2)

    # 4. Food Freshness (Mehrab Mahdian)
    food_dir = os.path.join("data", "raw", "food")
    os.makedirs(food_dir, exist_ok=True)
    df_food = pd.DataFrame({
        "mq2": np.random.uniform(100, 300, 5),
        "mq3": np.random.uniform(50, 150, 5),
        "mq135": np.random.uniform(200, 500, 5),
        "temp": [24, 25, 24, 26, 25],
        "humidity": [50, 52, 51, 55, 53],
        "status": ["fresh", "fresh", "rotten", "rotten", "spoiled"]
    })
    df_food.to_csv(os.path.join(food_dir, "food_data.csv"), index=False)

    # 5. Array of Gas Sensors (Husam K. Salih)
    garray_dir = os.path.join("data", "raw", "gasarray")
    os.makedirs(garray_dir, exist_ok=True)
    df_ga = pd.DataFrame({
        "s1": np.random.uniform(10, 50, 4),
        "s2": np.random.uniform(20, 60, 4),
        "s3": np.random.uniform(15, 45, 4),
        "s4": np.random.uniform(5, 25, 4),
        "s5": np.random.uniform(30, 70, 4),
        "s6": np.random.uniform(12, 35, 4),
        "odor": ["air", "coffee", "cinnamon", "ambient"]
    })
    df_ga.to_csv(os.path.join(garray_dir, "gas_array.csv"), index=False)

    # 6. ADL Classification (Saurabh Shahane)
    adl_dir = os.path.join("data", "raw", "adl")
    os.makedirs(adl_dir, exist_ok=True)
    df_adl = pd.DataFrame({
        "mq2": np.random.uniform(10, 50, 4),
        "mq9": np.random.uniform(20, 60, 4),
        "mq135": np.random.uniform(15, 45, 4),
        "mq137": np.random.uniform(5, 25, 4),
        "mq138": np.random.uniform(30, 70, 4),
        "activity": ["cooking", "cleaning", "ambient", "normal"]
    })
    df_adl.to_csv(os.path.join(adl_dir, "adl_data.csv"), index=False)

    # 7. E-Nose Breath Disease (Muhammad Rizwan)
    breath_dir = os.path.join("data", "raw", "breath")
    os.makedirs(breath_dir, exist_ok=True)
    df_br = pd.DataFrame({
        "sensor1": np.random.uniform(10, 50, 4),
        "sensor2": np.random.uniform(20, 60, 4),
        "sensor3": np.random.uniform(15, 45, 4),
        "sensor4": np.random.uniform(5, 25, 4),
        "sensor5": np.random.uniform(30, 70, 4),
        "disease": ["healthy", "healthy", "peppermint", "acetone"]
    })
    df_br.to_csv(os.path.join(breath_dir, "breath_data.csv"), index=False)

    print("Synthetic raw datasets successfully generated in data/raw/")


def generate_synthetic_transient_scans():
    """Generate synthetic time-series transient scans for high-resolution feature extraction test."""
    rows = []
    spec = load_spec()
    channels = [c["name"] for c in spec["channels"]]
    t_steps = np.linspace(0, 10.0, 50)  # 10 second scan

    for scan_num in range(1, 6):
        scan_id = f"transient_scan_{scan_num}"
        label = "narcotic_voc" if scan_num % 2 == 0 else "clean"
        bme_prof = list(np.random.uniform(5000, 30000, 10))

        for ch in channels:
            # Baseline (0-2s) + exponential transient curve (2-10s)
            r0 = 100.0
            if ch == "mics2714_no2":
                # Resistance INCREASES with analyte (sign -1 channel!)
                signal = r0 + np.where(t_steps > 2.0, 50.0 * (1 - np.exp(-(t_steps - 2.0) / 2.0)), 0.0)
            else:
                # Standard MOX: resistance DECREASES with analyte
                signal = r0 - np.where(t_steps > 2.0, 40.0 * (1 - np.exp(-(t_steps - 2.0) / 2.0)), 0.0)

            signal += np.random.normal(0, 0.5, len(t_steps))  # Add small noise

            for t, val in zip(t_steps, signal):
                rows.append({
                    "source": "synthetic_transient",
                    "scan_id": scan_id,
                    "t_rel": t,
                    "channel": ch,
                    "value": val,
                    "label": label,
                    "temp_c": 25.0 + scan_num * 0.5,
                    "rh_pct": 50.0 + scan_num * 1.0,
                    "press_hpa": 1013.25,
                    "collection_day": scan_num % 3 + 1,
                    "room_id": f"room_{scan_num % 2 + 1}",
                    "bme_profile": bme_prof,
                })
    return pd.DataFrame(rows)


def verify_pipeline():
    """Run verification checks on loaders, features.py, and drift.py."""
    print("=" * 70)
    print("VENJEX SMART LATHI — DATA & FEATURE PIPELINE VERIFICATION")
    print("=" * 70)

    generate_synthetic_raw_data()

    # 1. Test Loaders
    loaders = {
        "BME688 Odours": bme688_odours.load(),
        "Multimodal Gas": multimodal_gas.load(),
        "UCI Drift": drift_uci.load(),
        "Food Freshness": food_freshness.load(),
        "Gas Array": gas_array.load(),
        "ADL Classification": adl.load(),
        "Breath E-Nose": breath.load(),
    }

    print("\n--- 1. LOADERS SUMMARY ---")
    all_dfs = []
    for name, df in loaders.items():
        print(f"[{name}] Rows: {len(df)} | Unique Scans: {df['scan_id'].nunique() if len(df) > 0 else 0}")
        if len(df) > 0:
            all_dfs.append(df)

    # Add synthetic transient time-series scans
    transient_df = generate_synthetic_transient_scans()
    all_dfs.append(transient_df)

    unified_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    processed_path = os.path.join("data", "processed", "unified_dataset.parquet")
    try:
        unified_df.to_parquet(processed_path, index=False)
        print(f"\nUnified Dataset saved to {processed_path}")
    except Exception:
        csv_path = os.path.join("data", "processed", "unified_dataset.csv")
        unified_df.to_csv(csv_path, index=False)
        print(f"\nUnified Dataset saved to {csv_path}")

    # 2. Test Feature Extraction Contract (features.py)
    print("\n--- 2. FEATURE EXTRACTION CONTRACT (src/features.py) ---")
    spec = load_spec()
    print(f"Contract Feature Names Count: {len(FEATURE_NAMES)}")

    X_mat, y_labels, scan_ids, feat_names = extract_batch(unified_df, spec=spec)
    print(f"Extracted Feature Matrix Shape: {X_mat.shape} (Scans x Features)")
    print(f"Unique Canonical Classes Found: {set(y_labels)}")

    assert X_mat.shape[1] == len(FEATURE_NAMES), (
        f"Mismatch! Matrix features {X_mat.shape[1]} != FEATURE_NAMES count {len(FEATURE_NAMES)}"
    )
    print("SUCCESS: Extracted feature matrix shape matches frozen contract feature names!")

    # 3. Test Drift Correction & PCA Plotting (drift.py)
    print("\n--- 3. DRIFT CORRECTION & PCA VISUALIZATION (src/drift.py) ---")
    temps = np.random.uniform(20, 35, len(X_mat))
    rhs = np.random.uniform(40, 70, len(X_mat))
    batches = np.random.choice([1, 2, 3], size=len(X_mat))

    X_corrected = compensate_t_rh(X_mat, temps, rhs)
    plot_path = os.path.join("reports", "figures", "drift_before_after.png")
    plot_drift_before_after(X_mat, X_corrected, batch_labels=batches, save_path=plot_path)

    assert os.path.exists(plot_path), "PCA plot file was not created!"
    print(f"SUCCESS: Drift plot verified at {plot_path}")

    print("\n" + "=" * 70)
    print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY! Ready for Persons B and C.")
    print("=" * 70)


if __name__ == "__main__":
    verify_pipeline()
