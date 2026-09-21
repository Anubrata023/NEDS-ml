"""Config-driven feature extraction contract. Single source of truth for model input."""
import os
import yaml
import numpy as np
import pandas as pd

TRANSIENT_NAMES = ["ss", "slope", "auc", "rise90", "peak"]


def load_spec(path="config/sensors.yaml"):
    """Load sensor specification from YAML file."""
    if not os.path.exists(path):
        # Fallback search relative to workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "config", "sensors.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _transient_feats(t, v, baseline_s):
    """Compute 5 transient features from one channel's time series.
    
    Args:
        t: Relative time array (seconds)
        v: Sign-normalized signal values (v * sign)
        baseline_s: Duration of baseline period at start of scan
        
    Returns:
        [ss, slope, auc, rise, peak]
    """
    if len(v) == 0:
        return [np.nan] * len(TRANSIENT_NAMES)
    if len(v) == 1:
        return [float(v[0]), np.nan, np.nan, np.nan, float(v[0])]

    base = v[t <= baseline_s]
    r0 = float(np.median(base)) if len(base) > 0 else float(v[0])
    if abs(r0) < 1e-6:
        r0 = 1e-6 if r0 >= 0 else -1e-6

    # Normalized delta R / R0
    d = (v - r0) / abs(r0)

    # Steady state: median of last 20% of transient window
    ss_window = max(1, len(d) // 5)
    ss = float(np.median(d[-ss_window:]))

    # Slope: maximum rate of change
    if len(d) > 2 and (t[-1] - t[0]) > 0:
        dt = np.gradient(t)
        dt[dt == 0] = 1e-6
        grad = np.gradient(d) / dt
        slope = float(np.max(grad))
    else:
        slope = np.nan

    # AUC: Area under normalized response curve
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapz_fn(d, t)) if trapz_fn is not None else 0.0

    # Rise 90: time to reach 90% of steady state
    target = 0.9 * ss
    if ss > 0:
        idxs = np.where(d >= target)[0]
        rise = float(t[idxs[0]]) if len(idxs) > 0 else np.nan
    elif ss < 0:
        idxs = np.where(d <= target)[0]
        rise = float(t[idxs[0]]) if len(idxs) > 0 else np.nan
    else:
        rise = np.nan

    # Peak: maximum absolute response magnitude
    peak = float(np.max(d))

    return [ss, slope, auc, rise, peak]


def get_feature_names(spec=None):
    """Eagerly compute and return the frozen feature name list from spec."""
    if spec is None:
        spec = load_spec()

    names = []
    # 1. Analog channel transient features
    for c in spec["channels"]:
        for n in TRANSIENT_NAMES:
            names.append(f"{c['name']}_{n}")

    # 2. BME688 profile step features & step deltas
    n_bme = spec.get("bme688", {}).get("profile_steps", 10)
    for i in range(n_bme):
        names.append(f"bme_p{i}")
    for i in range(n_bme - 1):
        names.append(f"bme_d{i}")

    # 3. Cross-sensor ratios
    for a, b in spec.get("ratios", []):
        names.append(f"ratio_{a}_{b}")

    # 4. Environmental features
    for env_k in ["temp_c", "rh_pct", "press_hpa"]:
        names.append(env_k)

    return names


FEATURE_NAMES = get_feature_names()


def extract(scan_df, spec=None):
    """Extract feature vector for ONE scan_id.
    
    Args:
        scan_df: DataFrame containing long-format rows for a single scan_id.
                 Columns expected: ['channel', 't_rel', 'value', ...]
                 Optionally scan_df.attrs['bme_profile'] has BME 10-step profile.
        spec: Loaded sensor spec dict (or loaded from config if None).
        
    Returns:
        (feats_array, feature_names)
    """
    if spec is None:
        spec = load_spec()

    feats = []
    names = []
    per_chan_ss = {}

    baseline_s = spec.get("transient", {}).get("baseline_s", 2.0)

    # 1. Process standard analog channels
    for c in spec["channels"]:
        c_name = c["name"]
        sign = c.get("sign", 1)
        sub = scan_df[scan_df["channel"] == c_name].sort_values("t_rel")

        if len(sub) == 0:
            vals = [np.nan] * len(TRANSIENT_NAMES)
        else:
            t_vals = sub["t_rel"].values
            v_vals = sub["value"].values * sign  # CRITICAL: apply sign normalization FIRST
            vals = _transient_feats(t_vals, v_vals, baseline_s)

        per_chan_ss[c_name] = vals[0]
        feats.extend(vals)
        names.extend([f"{c_name}_{n}" for n in TRANSIENT_NAMES])

    # 2. Process BME688 10-step heater profile
    n_bme = spec.get("bme688", {}).get("profile_steps", 10)
    bme_profile = scan_df.attrs.get("bme_profile")

    if bme_profile is None:
        # Fallback: check if bme profile steps are passed as channels bme_p0..bme_p9
        bme_vals = []
        for i in range(n_bme):
            sub_bme = scan_df[scan_df["channel"] == f"bme_p{i}"]
            if len(sub_bme) > 0:
                bme_vals.append(float(sub_bme["value"].iloc[0]))
            else:
                bme_vals.append(np.nan)
        bme_profile = bme_vals

    bme_prof_arr = np.asarray(bme_profile, dtype=float)
    if len(bme_prof_arr) < n_bme:
        pad = np.full(n_bme - len(bme_prof_arr), np.nan)
        bme_prof_arr = np.concatenate([bme_prof_arr, pad])
    else:
        bme_prof_arr = bme_prof_arr[:n_bme]

    feats.extend(list(bme_prof_arr))
    names.extend([f"bme_p{i}" for i in range(n_bme)])

    bme_deltas = np.diff(bme_prof_arr)
    feats.extend(list(bme_deltas))
    names.extend([f"bme_d{i}" for i in range(n_bme - 1)])

    # 3. Cross-sensor ratios (steady-state ratios)
    for a, b in spec.get("ratios", []):
        va = per_chan_ss.get(a, np.nan)
        vb = per_chan_ss.get(b, np.nan)
        if pd.isna(va) or pd.isna(vb) or abs(vb) < 1e-6:
            r_val = np.nan
        else:
            r_val = float(va / (vb + 1e-6))
        feats.append(r_val)
        names.append(f"ratio_{a}_{b}")

    # 4. Environmental conditions
    for env_k in ["temp_c", "rh_pct", "press_hpa"]:
        if env_k in scan_df.columns and not scan_df[env_k].isna().all():
            val = float(scan_df[env_k].median())
        else:
            val = np.nan
        feats.append(val)
        names.append(env_k)

    return np.array(feats, dtype=float), names


def extract_batch(df_long, spec=None):
    """Extract features for multiple scans in a long-format DataFrame.
    
    Args:
        df_long: Long-format DataFrame containing multiple scan_ids.
        spec: Loaded sensor spec dict.
        
    Returns:
        (X_features_2d_array, y_labels_list, scan_ids_list, feature_names)
    """
    if spec is None:
        spec = load_spec()

    X_list = []
    y_list = []
    scan_id_list = []

    grouped = df_long.groupby("scan_id", sort=False)
    feat_names = None

    for scan_id, group in grouped:
        # Preserve attrs if present
        bme_prof = getattr(group, "attrs", {}).get("bme_profile")
        if bme_prof is None and "bme_profile" in group.columns:
            bme_prof = group["bme_profile"].dropna().values
            if len(bme_prof) > 0:
                bme_prof = bme_prof[0]
        if bme_prof is not None:
            group.attrs["bme_profile"] = bme_prof

        vec, feat_names = extract(group, spec=spec)
        X_list.append(vec)

        # Get scan label
        lbl = group["label"].iloc[0] if "label" in group.columns else "unknown"
        y_list.append(lbl)
        scan_id_list.append(scan_id)

    X_matrix = np.vstack(X_list) if len(X_list) > 0 else np.empty((0, len(FEATURE_NAMES)))
    return X_matrix, y_list, scan_id_list, feat_names
