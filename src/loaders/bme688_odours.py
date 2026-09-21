"""Loader for BME688 Odours Dataset (Laura Basa - P0).

Extracts 10-step heater profile resistance values across odor classes:
Air, Star Anise, Cinnamon, Cocoa, Coffee, Tea.
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical


def load(data_dir="data/raw/bme688"):
    """Load BME688 odours dataset into unified schema."""
    rows = []
    class_map = {
        "air": "clean",
        "star anise": "benign_odour",
        "cinnamon": "benign_odour",
        "cocoa": "benign_odour",
        "coffee": "benign_odour",
        "tea": "benign_odour",
    }

    if not os.path.exists(data_dir):
        return pd.DataFrame(columns=[
            "source", "scan_id", "t_rel", "channel", "value", "label",
            "temp_c", "rh_pct", "press_hpa", "collection_day", "room_id"
        ])

    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    for fpath in csv_files:
        filename = os.path.basename(fpath)
        odor_name = os.path.basename(os.path.dirname(fpath)).lower()
        if not odor_name or odor_name == "bme688":
            odor_name = filename.split("_")[0].lower()

        canonical_label = class_map.get(odor_name, map_to_canonical(odor_name))
        scan_id = f"bme_{os.path.splitext(filename)[0]}"

        try:
            df = pd.read_csv(fpath)
            # Standard BME format: columns heater_step, cycle_1 .. cycle_5
            cycle_cols = [c for c in df.columns if c.startswith("cycle_")]
            if len(cycle_cols) > 0 and "heater_step" in df.columns:
                # Take average across cycles for each heater step
                profile = df[cycle_cols].mean(axis=1).values[:10]
            else:
                # Fallback numeric parsing
                profile = df.iloc[:10, 1].values if df.shape[1] > 1 else df.iloc[:10, 0].values

            for step_idx, val in enumerate(profile):
                rows.append({
                    "source": "bme688_odours",
                    "scan_id": scan_id,
                    "t_rel": 0.0,
                    "channel": f"bme_p{step_idx}",
                    "value": float(val),
                    "label": canonical_label,
                    "temp_c": np.nan,
                    "rh_pct": np.nan,
                    "press_hpa": np.nan,
                    "collection_day": 1,
                    "room_id": "lab_1",
                })
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")

    return pd.DataFrame(rows)
