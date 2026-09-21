"""Loader for Food Freshness E-Nose Dataset (Mehrab Mahdian - P1 Pipeline Dress Rehearsal).

Uses MQ sensors (MQ2, MQ3, MQ4, MQ5, MQ6, MQ7, MQ8, MQ9, MQ135).
Labels:
Fresh -> clean
Rotten / spoiled -> benign_odour (hard negative)
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical


def load(data_dir="data/raw/food"):
    """Load Food Freshness dataset into unified schema."""
    rows = []
    class_map = {
        "fresh": "clean",
        "good": "clean",
        "rotten": "benign_odour",
        "spoiled": "benign_odour",
        "bad": "benign_odour",
    }

    if not os.path.exists(data_dir):
        return pd.DataFrame(columns=[
            "source", "scan_id", "t_rel", "channel", "value", "label",
            "temp_c", "rh_pct", "press_hpa", "collection_day", "room_id"
        ])

    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            sensor_cols = [c for c in df.columns if any(mq in c.lower() for mq in ["mq", "sensor"])]
            if len(sensor_cols) == 0:
                sensor_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]][:8]

            label_col = [c for c in df.columns if any(l in c.lower() for l in ["label", "state", "status", "class", "target"])]
            label_name = label_col[0] if len(label_col) > 0 else df.columns[-1]

            chan_names = ["mq2", "mq3", "mq135", "mq137", "mq138", "tgs2620", "mics2714_no2", "mics2714_h2"]

            for idx, r in df.iterrows():
                scan_id = f"food_{idx}"
                raw_lbl = str(r[label_name]).strip().lower()
                canonical_lbl = class_map.get(raw_lbl, map_to_canonical(raw_lbl))

                temp = float(r["temp"]) if "temp" in r else np.nan
                rh = float(r["humidity"]) if "humidity" in r else np.nan

                for s_idx, s_col in enumerate(sensor_cols):
                    ch_name = chan_names[s_idx] if s_idx < len(chan_names) else f"mq_{s_idx}"
                    rows.append({
                        "source": "food_freshness",
                        "scan_id": scan_id,
                        "t_rel": 0.0,
                        "channel": ch_name,
                        "value": float(r[s_col]),
                        "label": canonical_lbl,
                        "temp_c": temp,
                        "rh_pct": rh,
                        "press_hpa": np.nan,
                        "collection_day": 1,
                        "room_id": "lab_1",
                    })
        except Exception as e:
            print(f"Warning: Failed to load food freshness file {fpath}: {e}")

    return pd.DataFrame(rows)
