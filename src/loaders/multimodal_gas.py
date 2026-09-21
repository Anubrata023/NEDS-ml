"""Loader for Multimodal Gas Detection Dataset (JOAI Competition - P0 Hard Negatives).

Contains 7 gas sensors + hard negative classes:
No Gas -> clean
Perfume, Smoke, Mixture -> benign_odour
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical


def load(data_dir="data/raw/multimodal"):
    """Load Multimodal Gas dataset into unified schema."""
    rows = []
    class_map = {
        "nogas": "clean",
        "no gas": "clean",
        "clean": "clean",
        "perfume": "benign_odour",
        "smoke": "benign_odour",
        "mixture": "benign_odour",
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
            # Find sensor columns (usually 7 sensor columns)
            sensor_cols = [c for c in df.columns if "sensor" in c.lower() or "s" in c.lower() and c.lower() != "serial_no"]
            if len(sensor_cols) == 0:
                sensor_cols = [c for c in df.columns[1:8]]

            label_col = [c for c in df.columns if "class" in c.lower() or "label" in c.lower() or "target" in c.lower()]
            label_col_name = label_col[0] if len(label_col) > 0 else df.columns[-1]

            # Standard analog channel names mapping
            chan_names = ["mq2", "mq3", "mq135", "mq137", "mq138", "mics2714_no2", "mics2714_h2"]

            for idx, r in df.iterrows():
                serial = r.get("serial_no", idx)
                scan_id = f"mm_{serial}"
                raw_lbl = str(r[label_col_name]).strip().lower()
                canonical_lbl = class_map.get(raw_lbl, map_to_canonical(raw_lbl))

                for s_idx, s_col in enumerate(sensor_cols[:7]):
                    ch_name = chan_names[s_idx] if s_idx < len(chan_names) else f"sensor_{s_idx+1}"
                    rows.append({
                        "source": "multimodal_gas",
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
            print(f"Warning: Failed to load {fpath}: {e}")

    return pd.DataFrame(rows)
