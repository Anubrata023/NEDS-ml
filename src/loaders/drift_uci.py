"""Loader for UCI Gas Sensor Array Drift Dataset (Vergara - P0).

Contains 16 chemical sensors exposed to 6 gases across 10 batches (36 months).
Gases:
1: Ethanol (narcotic_voc proxy)
2: Ethylene (benign_odour)
3: Ammonia (nitrate_trace proxy)
4: Acetaldehyde (narcotic_voc proxy)
5: Acetone (narcotic_voc proxy)
6: Toluene (narcotic_voc proxy)
"""
import glob
import os
import numpy as np
import pandas as pd
from src.loaders import map_to_canonical

GAS_CLASS_MAP = {
    1: "narcotic_voc",   # Ethanol
    2: "benign_odour",   # Ethylene
    3: "nitrate_trace",  # Ammonia
    4: "narcotic_voc",   # Acetaldehyde
    5: "narcotic_voc",   # Acetone
    6: "narcotic_voc",   # Toluene
}

# Standard mapping of the 16 Figaro TGS sensors to canonical channels
SENSOR_CHANNEL_MAP = [
    "mq2", "mq3", "mq135", "mq137", "mq138", "tgs2620", "mics2714_no2", "mics2714_h2",
    "tgs2600_1", "tgs2602_1", "tgs2610_1", "tgs2611_1", "tgs2612_1", "tgs2620_2", "tgs822_1", "tgs822_2"
]


def load(data_dir="data/raw/drift"):
    """Load UCI Drift dataset into unified schema."""
    rows = []
    if not os.path.exists(data_dir):
        return pd.DataFrame(columns=[
            "source", "scan_id", "t_rel", "channel", "value", "label",
            "temp_c", "rh_pct", "press_hpa", "collection_day", "room_id"
        ])

    batch_files = glob.glob(os.path.join(data_dir, "**", "batch*.dat"), recursive=True) + \
                  glob.glob(os.path.join(data_dir, "**", "batch*.csv"), recursive=True)

    for fpath in batch_files:
        filename = os.path.basename(fpath)
        batch_num = 1
        try:
            # Extract batch number from filename e.g. batch1.dat -> 1
            digits = "".join(filter(str.isdigit, filename))
            if digits:
                batch_num = int(digits)
        except Exception:
            batch_num = 1

        try:
            # LIBSVM format or CSV format
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    gas_label_id = int(parts[0].split(";")[0])
                except Exception:
                    continue

                canonical_label = GAS_CLASS_MAP.get(gas_label_id, "benign_odour")
                scan_id = f"uci_b{batch_num}_s{idx}"

                # Parse feature index:value pairs
                feats_dict = {}
                for p in parts[1:]:
                    if ":" in p:
                        fid, fval = p.split(":", 1)
                        feats_dict[int(fid)] = float(fval)

                # Extract steady-state feature (usually feature ID 1..16 or modulo 16)
                for s_idx in range(16):
                    # In UCI dataset, feature 1 + s_idx * 8 is the primary steady state Delta R feature
                    feat_id = (s_idx * 8) + 1
                    val = feats_dict.get(feat_id, feats_dict.get(s_idx + 1, 0.0))
                    ch_name = SENSOR_CHANNEL_MAP[s_idx]

                    rows.append({
                        "source": "drift_uci",
                        "scan_id": scan_id,
                        "t_rel": 0.0,
                        "channel": ch_name,
                        "value": float(val),
                        "label": canonical_label,
                        "temp_c": np.nan,
                        "rh_pct": np.nan,
                        "press_hpa": np.nan,
                        "collection_day": batch_num,  # Batch number represents time horizon!
                        "room_id": f"batch_{batch_num}",
                    })
        except Exception as e:
            print(f"Warning: Failed to load UCI batch {fpath}: {e}")

    return pd.DataFrame(rows)
