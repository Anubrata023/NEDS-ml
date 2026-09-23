"""Drift correction and baseline re-estimation module for Venjex Smart Lathi.

Provides:
1. Baseline normalization (Delta R / R0)
2. Rolling ambient re-baseline for continuous operation
3. Environmental (T/RH) compensation via least-squares baseline fit
4. PCA before/after drift visualization plotting function for reports
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def baseline_normalize(v, r0=None):
    """Normalize raw sensor signal by baseline R0: (v - R0) / R0."""
    v_arr = np.asarray(v, dtype=float)
    if r0 is None or r0 == 0:
        r0 = np.median(v_arr[:max(1, len(v_arr) // 5)]) if len(v_arr) > 0 else 1.0
    if abs(r0) < 1e-6:
        r0 = 1e-6 if r0 >= 0 else -1e-6
    return (v_arr - r0) / abs(r0)


def rebaseline_rolling(v_series, window_samples=180):
    """Rolling ambient re-baselining for continuous live monitoring mode.
    
    Args:
        v_series: Time-series of raw sensor readings
        window_samples: Rolling window size (e.g. 180 samples = 3 minutes at 1Hz)
        
    Returns:
        Re-baselined signal array
    """
    v_arr = np.asarray(v_series, dtype=float)
    if len(v_arr) < 2:
        return v_arr
    rolling_r0 = pd.Series(v_arr).rolling(window=window_samples, min_periods=1).median().values
    rolling_r0[rolling_r0 == 0] = 1e-6
    return (v_arr - rolling_r0) / np.abs(rolling_r0)


def compensate_t_rh(X, temp, rh, is_clean_mask=None, coef=None, return_coef=False):
    """Temperature and Relative Humidity baseline compensation.
    
    Fits value ~ a*T + b*RH + c on clean-air scans only (or all if mask is None),
    and subtracts the fitted environmental component from all scans.
    
    Args:
        X: Feature matrix of shape (N_samples, N_features)
        temp: Temperature vector of shape (N_samples,)
        rh: Relative humidity vector of shape (N_samples,)
        is_clean_mask: Boolean mask indicating clean-air scans for fitting.
        coef: Optional pre-fitted coefficients (3 x N_features). If provided, applies transform directly.
        return_coef: If True, returns (X_corrected, coef) tuple.
        
    Returns:
        X_corrected: Environmentally compensated feature matrix (or (X_corrected, coef) if return_coef=True)
    """
    X_mat = np.array(X, dtype=float, copy=True)
    temp_vec = np.array(temp, dtype=float, copy=True)
    rh_vec = np.array(rh, dtype=float, copy=True)

    # Impute NaNs in temp/rh with median
    if np.isnan(temp_vec).all():
        temp_vec = np.full_like(temp_vec, 25.0)
    else:
        temp_vec[np.isnan(temp_vec)] = np.nanmedian(temp_vec)

    if np.isnan(rh_vec).all():
        rh_vec = np.full_like(rh_vec, 50.0)
    else:
        rh_vec[np.isnan(rh_vec)] = np.nanmedian(rh_vec)

    # Build design matrix A = [T, RH, 1]
    A = np.column_stack([temp_vec, rh_vec, np.ones_like(temp_vec)])
    X_mat_clean = np.nan_to_num(X_mat, nan=0.0)

    # If pre-fitted coefficients provided, apply directly
    if coef is not None:
        env_trend = A[:, :2] @ coef[:2, :]
        X_corrected = X_mat_clean - env_trend
        return (X_corrected, coef) if return_coef else X_corrected

    # If single sample or underdetermined without coef, skip fitting
    if len(X_mat) < 3:
        zero_coef = np.zeros((3, X_mat.shape[1]))
        return (X_mat_clean, zero_coef) if return_coef else X_mat_clean

    if is_clean_mask is None or not np.any(is_clean_mask):
        is_clean_mask = np.ones(len(X_mat), dtype=bool)

    A_fit = A[is_clean_mask]
    X_fit = X_mat[is_clean_mask]

    # Handle NaNs in X for fitting
    X_fit_clean = np.nan_to_num(X_fit, nan=0.0)

    # Fit linear regression per feature
    coef, *_ = np.linalg.lstsq(A_fit, X_fit_clean, rcond=None)

    # Subtract environmental effect (omit intercept subtraction to preserve mean)
    env_trend = A[:, :2] @ coef[:2, :]
    X_corrected = X_mat_clean - env_trend

    return (X_corrected, coef) if return_coef else X_corrected


def plot_drift_before_after(X_raw, X_corrected, batch_labels, save_path="reports/figures/drift_before_after.png"):
    """Generate side-by-side PCA plot of sensor features before vs after drift correction.
    
    Args:
        X_raw: Uncorrected feature matrix (N x P)
        X_corrected: Drift/T-RH corrected feature matrix (N x P)
        batch_labels: Vector of batch/day identifiers
        save_path: File path to save output PNG
    """
    # Impute NaNs for PCA
    X_raw_clean = np.nan_to_num(X_raw, nan=0.0)
    X_corr_clean = np.nan_to_num(X_corrected, nan=0.0)

    # Fit PCA
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    pca1 = PCA(n_components=2).fit_transform(scaler1.fit_transform(X_raw_clean))
    pca2 = PCA(n_components=2).fit_transform(scaler2.fit_transform(X_corr_clean))

    unique_batches = np.unique(batch_labels)
    if hasattr(plt, "colormaps"):
        cmap = plt.colormaps["viridis"].resampled(max(1, len(unique_batches)))
    else:
        cmap = plt.cm.get_cmap("viridis", len(unique_batches))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for idx, b in enumerate(unique_batches):
        mask = (batch_labels == b)
        color = cmap(idx)
        ax1.scatter(pca1[mask, 0], pca1[mask, 1], label=f"Batch/Day {b}", alpha=0.7, color=color)
        ax2.scatter(pca2[mask, 0], pca2[mask, 1], label=f"Batch/Day {b}", alpha=0.7, color=color)

    ax1.set_title("BEFORE Drift Correction (Raw Data)")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title("AFTER Drift & T/RH Correction")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Venjex Smart Lathi — MOX Sensor Drift Compensation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Drift correction plot successfully saved to {save_path}")
