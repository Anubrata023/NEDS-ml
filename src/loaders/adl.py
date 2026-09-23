"""Loader for ADL Classification Dataset (Saurabh Shahane - P2 False Positive Negatives).

Contains MQ2, MQ9, MQ135, MQ137, MQ138 responses for daily living activities
(cooking, cleaning, household odors).
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical


def load(data_dir="data/raw/adl"):
    """Load ADL Air Quality classification dataset into unified schema."""
    rows = []
    class_map = {
        "cooking": "benign_odour",
        "cleaning": "benign_odour",
        "normal": "clean",
        "background": "clean",
        "ambient": "clean",
    }

    if not os.path.exists(data_dir):
        return pd.DataFrame(columns=[
            "source", "scan_id", "t_rel", "channel", "value", "label",
            "temp_c", "rh_pct", "press_hpa", "collection_day", "room_id"
        ])

    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    chan_names = ["mq2", "mq3", "mq135", "mq137", "mq138", "tgs2620"]

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            sensor_cols = [c for c in df.columns if any(mq in c.lower() for mq in ["mq", "sensor"])]
            if len(sensor_cols) == 0:
                sensor_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]][:5]

            label_cols = [c for c in df.columns if any(l in c.lower() for l in ["label", "class", "activity"])]
            label_name = label_cols[0] if len(label_cols) > 0 else df.columns[-1]

            for idx, r in df.iterrows():
                scan_id = f"adl_{idx}"
                raw_lbl = str(r[label_name]).strip().lower()
                canonical_lbl = class_map.get(raw_lbl, map_to_canonical(raw_lbl))

                for s_idx, s_col in enumerate(sensor_cols):
                    ch_name = chan_names[s_idx] if s_idx < len(chan_names) else f"mq_{s_idx}"
                    rows.append({
                        "source": "adl_classification",
                        "scan_id": scan_id,
                        "t_rel": 0.0,
                        "channel": ch_name,
                        "value": float(r[s_col]),
                        "label": canonical_lbl,
                        "temp_c": np.nan,
                        "rh_pct": np.nan,
                        "press_hpa": np.nan,
                        "collection_day": 1,
                        "room_id": "lab_1",
                    })
        except Exception as e:
            print(f"Warning: Failed to load ADL file {fpath}: {e}")

    return pd.DataFrame(rows)
