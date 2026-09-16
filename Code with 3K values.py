# -*- coding: utf-8 -*-
"""
Updated paper_17_ieee.py
Aligned with X-HealthGuard Methodology (Version 1.0)

CHANGES vs original version (all genuine algorithmic/methodological
improvements -- no hardcoded/fabricated metrics):

  1. Real SampEn (Sample Entropy) and DFA alpha1 (Detrended Fluctuation
     Analysis) computed in pure NumPy, replacing the old hardcoded
     placeholders (1.5 / 1.0). Those placeholders made ~1/8 of every
     256-D feature vector constant and useless for the classifier.
  2. Fixed a labeling bug: non-beat annotation symbols in MIT-BIH
     ('+', '~', '|', '"', 'x', etc.) were being counted as "anomaly"
     even though they are not abnormal heartbeats. This was injecting
     label noise that directly hurt precision.
  3. Uses the FULL 48-record MIT-BIH Arrhythmia Database by default
     instead of a 7-record subset -- small subsets give unstable,
     unreliable F1/precision/recall estimates.
  4. Added StandardScaler (fit on train fold only, applied to test).
  5. Added RandomizedSearchCV hyperparameter search for the Random
     Forest instead of fixed guessed hyperparameters.
  6. Added probability-threshold tuning on the training fold using the
     precision-recall curve (a standard, legitimate technique) so the
     operating point can be chosen for a target precision/recall
     balance instead of the default 0.5 cutoff.
  7. SMOTE is still fit on the ACTIVE TRAINING FOLD ONLY (no leakage
     into test set), and splitting is still done at the record
     (patient) level via GroupShuffleSplit.

NOTE: This script does not, and cannot, guarantee an exact
Precision/Recall/F1/Accuracy number -- that depends on your actual
data. What it does is remove sources of noise/leakage/placeholder
features that were suppressing performance, and add proper tuning, so
the numbers you get are the genuine best the pipeline can produce.
"""

import numpy as np
import pandas as pd
import wfdb
import os
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from biosppy.signals import ecg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, confusion_matrix,
    accuracy_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from scipy.signal import welch, butter, filtfilt
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# MIT-BIH non-beat / non-diagnostic annotation symbols.
# These must NOT be treated as "abnormal beat" anomalies.
NON_BEAT_SYMBOLS = {'+', '~', '|', '"', 'x', '[', ']', '!', 'U', '?'}

# =================================================================================
# PREPROCESSING UTILITIES
# =================================================================================
def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=360.0, order=4):
    """Zero-phase 4th-order Butterworth band-pass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def sample_entropy(rr, m=2, r_ratio=0.2):
    """
    Real Sample Entropy (SampEn) implementation, pure NumPy.
    rr        : 1D array of RR intervals
    m         : embedding dimension
    r_ratio   : tolerance as a fraction of std(rr)
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    if n < m + 2:
        return 0.0

    r = r_ratio * np.std(rr)
    if r == 0:
        return 0.0

    def _phi(m_):
        templates = np.array([rr[i:i + m_] for i in range(n - m_ + 1)])
        count = 0
        total = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1) if i + 1 < len(templates) else np.array([])
            if len(dist) > 0:
                count += np.sum(dist <= r)
            total += len(templates) - i - 1
        return count, total

    count_m, _ = _phi(m)
    count_m1, _ = _phi(m + 1)

    if count_m == 0 or count_m1 == 0:
        return 0.0
    return -np.log(count_m1 / count_m)


def dfa_alpha1(rr, min_box=4, max_box=16):
    """
    Real Detrended Fluctuation Analysis (short-term alpha1), pure NumPy.
    rr : 1D array of RR intervals (ms)
    """
    rr = np.asarray(rr, dtype=float)
    n = len(rr)
    if n < max_box * 2:
        return 1.0  # not enough data for a reliable estimate

    y = np.cumsum(rr - np.mean(rr))
    box_sizes = np.arange(min_box, min(max_box, n // 4) + 1)
    if len(box_sizes) < 2:
        return 1.0

    fluctuations = []
    for box in box_sizes:
        n_boxes = n // box
        if n_boxes < 1:
            continue
        rms_vals = []
        for i in range(n_boxes):
            seg = y[i * box:(i + 1) * box]
            x_axis = np.arange(box)
            coeffs = np.polyfit(x_axis, seg, 1)
            trend = np.polyval(coeffs, x_axis)
            rms_vals.append(np.sqrt(np.mean((seg - trend) ** 2)))
        fluctuations.append(np.mean(rms_vals))

    fluctuations = np.array(fluctuations)
    valid = fluctuations > 0
    if np.sum(valid) < 2:
        return 1.0

    log_box = np.log(box_sizes[valid])
    log_fluc = np.log(fluctuations[valid])
    alpha1 = np.polyfit(log_box, log_fluc, 1)[0]
    return alpha1


def calculate_nonlinear_features(rr_intervals):
    """Real non-linear features: SD1, SD2, SampEn, DFA alpha1."""
    if len(rr_intervals) < 4:
        return 0.0, 0.0, 0.0, 1.0
    sd1 = np.std(np.diff(rr_intervals)) / np.sqrt(2)
    sd2 = np.std(rr_intervals) * np.sqrt(2)
    sampen = sample_entropy(rr_intervals)
    alpha1 = dfa_alpha1(rr_intervals)
    return sd1, sd2, sampen, alpha1


# =================================================================================
# PART 1: X-HEALTHGUARD (RANDOM FOREST + 256D HRV)
# =================================================================================
def extract_16_hrv_features(window_rr, fs):
    """Extracts exactly 16 feature families for a given local window."""
    if len(window_rr) < 4:
        return np.zeros(16)

    # Time-domain (5 features)
    mean_rr = np.mean(window_rr)
    sdnn = np.std(window_rr)
    rr_diff = np.diff(window_rr)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
    sdsd = np.std(rr_diff) if len(rr_diff) > 0 else 0

    # Frequency-domain (7 features) - Interpolated at 4Hz
    total_power, vlf, lf, hf, lf_hf, lf_norm, hf_norm = 0, 0, 0, 0, 0, 0, 0
    if len(window_rr) > 4:
        try:
            rr_times = np.cumsum(window_rr) / 1000.0
            rr_times -= rr_times[0]
            f_interp = interp1d(rr_times, window_rr, kind='linear', fill_value='extrapolate')
            t_interp = np.arange(0, rr_times[-1], 1 / 4.0)
            rr_interp = f_interp(t_interp)

            freqs, psd = welch(rr_interp - np.mean(rr_interp), fs=4.0, nperseg=min(len(rr_interp), 256))

            vlf = np.trapz(psd[(freqs >= 0.003) & (freqs < 0.04)], freqs[(freqs >= 0.003) & (freqs < 0.04)])
            lf = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)], freqs[(freqs >= 0.04) & (freqs < 0.15)])
            hf = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)], freqs[(freqs >= 0.15) & (freqs < 0.40)])

            total_power = vlf + lf + hf
            lf_hf = lf / (hf + 1e-8)
            lf_norm = lf / (lf + hf + 1e-8)
            hf_norm = hf / (lf + hf + 1e-8)
        except Exception:
            pass

    # Non-linear (4 features) -- now REAL, not placeholders
    sd1, sd2, sampen, alpha1 = calculate_nonlinear_features(window_rr)

    return np.array([mean_rr, sdnn, rmssd, pnn50, sdsd, total_power, vlf, lf, hf,
                      lf_hf, lf_norm, hf_norm, sd1, sd2, sampen, alpha1])


def process_ecg_xhealthguard(records, fs=360):
    """Processes 30s segments into 16x16 = 256D feature vectors."""
    window_size_30s = 30 * fs
    local_window_15s = 15 * fs
    stride_1s = 1 * fs

    all_features, all_labels, all_groups = [], [], []
    db_dir = 'mitdb_data'

    for record_name in records:
        record_path = os.path.join(db_dir, record_name)
        if not os.path.exists(record_path + '.dat'):
            continue

        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"  Skipping {record_name}: {e}")
            continue

        signal = butter_bandpass_filter(record.p_signal[:, 0], fs=fs)

        try:
            ecg_data = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
            all_r_peaks = ecg_data['rpeaks']
        except Exception as e:
            print(f"  R-peak detection failed for {record_name}: {e}")
            continue

        for i in range(0, len(signal) - window_size_30s, window_size_30s):
            segment_start, segment_end = i, i + window_size_30s

            # FIX: exclude non-beat annotation symbols from anomaly labeling
            window_anns = [
                sym for loc, sym in zip(annotation.sample, annotation.symbol)
                if segment_start <= loc < segment_end and sym not in NON_BEAT_SYMBOLS
            ]
            is_anomaly = any(sym != 'N' for sym in window_anns)

            segment_256_vector = []
            for w in range(16):
                lw_start = segment_start + (w * stride_1s)
                lw_end = lw_start + local_window_15s

                peaks_in_lw = all_r_peaks[(all_r_peaks >= lw_start) & (all_r_peaks < lw_end)]
                rr_intervals = np.diff(peaks_in_lw) * (1000.0 / fs)
                rr_intervals = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]

                features_16 = extract_16_hrv_features(rr_intervals, fs)
                segment_256_vector.extend(features_16)

            all_features.append(segment_256_vector)
            all_labels.append(1 if is_anomaly else 0)
            all_groups.append(record_name)

    return np.array(all_features), np.array(all_labels), np.array(all_groups)


def tune_threshold(y_true, y_proba, target_precision=None, target_recall=None):
    """
    Pick an operating threshold from the precision-recall curve.
    If target_precision is given, picks the lowest threshold that still
    meets it (maximizing recall). Otherwise picks the threshold that
    maximizes F1. This is standard, legitimate threshold calibration --
    it does not alter the model's underlying predictions/scores.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    if target_precision is not None:
        valid = precisions >= target_precision
        if np.any(valid):
            idx = np.argmax(recalls[valid])
            return thresholds[valid][idx]

    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx]


def run_xhealthguard_experiment(target_precision=None):
    print("\n--- Running X-HealthGuard (Random Forest) Pipeline ---")

    db_dir = 'mitdb_data'
    record_files = glob.glob(f"{db_dir}/*.dat")
    RECORDS = [os.path.basename(f).replace('.dat', '') for f in record_files]

    if not RECORDS:
        print("Error: No records found in 'mitdb_data'. Please download the dataset first.")
        return

    if len(RECORDS) < 20:
        print(f"Warning: running on only {len(RECORDS)} records. "
              f"For stable, paper-level metrics, use the full 48-record MIT-BIH set.")

    print(f"Total records being processed: {len(RECORDS)}")

    # 1. Feature Extraction
    X, y, groups = process_ecg_xhealthguard(RECORDS)
    if len(X) == 0:
        print("Error: Extracted feature array is empty.")
        return
    print(f"Extracted {X.shape[0]} segments, {X.shape[1]}-D features. "
          f"Positive class ratio: {np.mean(y):.3f}")

    # 2. Record-Level Partitioning (80:20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 3. Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. SMOTE on Active Training Fold ONLY
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # 5. Hyperparameter tuning via RandomizedSearchCV
    param_dist = {
        'n_estimators': [200, 300, 400, 500, 600],
        'max_depth': [10, 15, 20, 30, None],
        'min_samples_split': [2, 4, 6, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_rf, param_distributions=param_dist, n_iter=25,
        scoring='f1', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train_bal, y_train_bal)
    rf = search.best_estimator_
    print(f"Best RF params: {search.best_params_}")

    # 6. Threshold tuning (on training fold predictions, NOT test)
    train_proba = rf.predict_proba(X_train_bal)[:, 1]
    threshold = tune_threshold(y_train_bal, train_proba, target_precision=target_precision)
    print(f"Selected decision threshold: {threshold:.3f}")

    # 7. Evaluation on held-out test set
    y_proba_test = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_test >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba_test)
    except Exception:
        auc = float('nan')

    print(f"X-HealthGuard (RF) Results -> Precision: {prec:.3f}, Recall: {rec:.3f}, "
          f"F1: {f1:.3f}, Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "auc": auc}


if __name__ == "__main__":
    # Full standard MIT-BIH Arrhythmia Database record list (48 records).
    FULL_MITDB_RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]

    print("Downloading full MIT-BIH Arrhythmia Database (48 records) from PhysioNet...")
    try:
        wfdb.dl_database('mitdb', dl_dir='mitdb_data', records=FULL_MITDB_RECORDS)
        print("Download complete!")
    except Exception as e:
        print(f"Download warning (can be ignored if files already exist): {e}")

    # target_precision=None -> threshold chosen to maximize F1.
    # Set e.g. target_precision=0.73 if you want to explicitly favor a
    # precision floor and let recall/F1 adjust accordingly.
    run_xhealthguard_experiment(target_precision=None)
