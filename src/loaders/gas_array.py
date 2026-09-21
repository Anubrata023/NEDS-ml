"""Loader for Array of Gas Sensors Dataset (Husam K. Salih - P1).

Contains data from 6 gas sensors for odor recognition.
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical


def load(data_dir="data/raw/gasarray"):
    """Load Gas Sensor Array dataset into unified schema."""
    rows = []
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
            sensor_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]][:6]
            label_cols = [c for c in df.columns if any(l in c.lower() for l in ["label", "class", "target", "odor"])]
            label_name = label_cols[0] if len(label_cols) > 0 else df.columns[-1]

            for idx, r in df.iterrows():
                scan_id = f"garray_{idx}"
                raw_lbl = str(r[label_name]).strip().lower()
                canonical_lbl = map_to_canonical(raw_lbl)

                for s_idx, s_col in enumerate(sensor_cols):
                    ch_name = chan_names[s_idx] if s_idx < len(chan_names) else f"sensor_{s_idx}"
                    rows.append({
                        "source": "gas_array",
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
            print(f"Warning: Failed to load gas array file {fpath}: {e}")

    return pd.DataFrame(rows)
