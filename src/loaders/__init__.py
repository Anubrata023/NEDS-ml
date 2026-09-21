"""Dataset loaders module for Venjex Smart Lathi pipeline.

All loaders return DataFrames strictly adhering to the unified long-format schema:
Columns: ['source', 'scan_id', 't_rel', 'channel', 'value', 'label', 'temp_c', 'rh_pct', 'press_hpa', 'collection_day', 'room_id']
"""
import os
import yaml
import pandas as pd

UNIFIED_COLUMNS = [
    "source",
    "scan_id",
    "t_rel",
    "channel",
    "value",
    "label",
    "temp_c",
    "rh_pct",
    "press_hpa",
    "collection_day",
    "room_id",
]


def load_class_mapping(path="config/classes.yaml"):
    """Load canonical class mapping dictionary."""
    if not os.path.exists(path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, "config", "classes.yaml")
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            mapping = {}
            for canonical_cls, raw_labels in cfg.get("canonical", {}).items():
                for raw_lbl in raw_labels:
                    mapping[raw_lbl.lower()] = canonical_cls
            return mapping
    return {}


def map_to_canonical(label, mapping=None):
    """Map a raw dataset label to canonical class."""
    if mapping is None:
        mapping = load_class_mapping()
    lbl_str = str(label).strip().lower()
    return mapping.get(lbl_str, "benign_odour")
