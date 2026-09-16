
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


def tune_threshold(y_true, y_proba, target_precision=None, objective='f1'):
    """
    Pick an operating threshold from the precision-recall curve.

    IMPORTANT: this must be called on a set whose class balance matches
    the REAL-WORLD / test-set distribution (i.e. NOT the SMOTE-balanced
    training data). Tuning on balanced data picks a threshold optimized
    for a 50:50 class ratio, which -- when applied to an imbalanced test
    set -- causes exactly the symptom of very high Recall but poor
    Accuracy (too many false positives on the large "normal" class).

    objective:
      'f1'       -> maximize F1
      'accuracy' -> maximize Accuracy directly
    If target_precision is given, it overrides objective and picks the
    lowest threshold that still meets that precision (maximizing recall
    subject to that constraint).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions, recalls = precisions[:-1], recalls[:-1]

    if target_precision is not None:
        valid = precisions >= target_precision
        if np.any(valid):
            idx = np.argmax(recalls[valid])
            return thresholds[valid][idx]

    if objective == 'accuracy':
        best_thr, best_acc = 0.5, -1
        for thr in thresholds:
            preds = (y_proba >= thr).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        return best_thr

    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx]


def run_xhealthguard_experiment(target_precision=None, threshold_objective='f1'):
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

    # 2. Record-Level Partitioning: Train+Val (80%) / Test (20%)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(gss_outer.split(X, y, groups))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    groups_trainval = groups[trainval_idx]

    # 2b. Further split Train+Val into a SMOTE-training subset and a
    #     clean, NATURALLY-IMBALANCED validation subset (record-level,
    #     no leakage). This validation subset mirrors the real-world
    #     class distribution -- same as the test set -- so tuning on it
    #     (instead of on SMOTE-balanced data) gives a threshold that
    #     generalizes correctly and fixes the high-Recall/low-Accuracy
    #     mismatch.
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    sub_train_idx, val_idx = next(gss_inner.split(X_trainval, y_trainval, groups_trainval))
    X_sub_train, X_val = X_trainval[sub_train_idx], X_trainval[val_idx]
    y_sub_train, y_val = y_trainval[sub_train_idx], y_trainval[val_idx]
    print(f"Sub-train segments: {len(X_sub_train)} (pos ratio {np.mean(y_sub_train):.3f}) | "
          f"Validation segments: {len(X_val)} (pos ratio {np.mean(y_val):.3f}, "
          f"naturally imbalanced -- matches test distribution)")

    # 3. Scale features (fit on Train+Val only, applied everywhere)
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_sub_train = scaler.transform(X_sub_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. SMOTE -- fit ONLY on the sub-train fold (val and test stay
    #    untouched and imbalanced, exactly as real data would be)
    smote = SMOTE(random_state=42)
    X_sub_train_bal, y_sub_train_bal = smote.fit_resample(X_sub_train, y_sub_train)

    # 5. Hyperparameter tuning via RandomizedSearchCV on the balanced
    #    sub-train fold
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
    search.fit(X_sub_train_bal, y_sub_train_bal)
    best_params = search.best_params_
    print(f"Best RF params: {best_params}")

    # 6. Threshold tuning on the CLEAN, naturally-imbalanced validation
    #    set (not on SMOTE-balanced data) -- this is the key fix.
    probe_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    probe_rf.fit(X_sub_train_bal, y_sub_train_bal)
    val_proba = probe_rf.predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_proba, target_precision=target_precision,
                                objective=threshold_objective)
    print(f"Selected decision threshold (tuned on realistic-imbalance validation set): {threshold:.3f}")

    # 7. Refit final model on ALL available training data (sub-train +
    #    val, SMOTE-balanced) so no data is wasted, then apply the
    #    tuned threshold to the untouched test set.
    X_trainval_bal, y_trainval_bal = smote.fit_resample(X_trainval_scaled, y_trainval)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    rf.fit(X_trainval_bal, y_trainval_bal)

    # 8. Evaluation on held-out test set
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

    # threshold_objective='f1'       -> best balance of precision/recall
    # threshold_objective='accuracy' -> directly maximizes accuracy
    # target_precision=0.XX          -> overrides objective; guarantees
    #                                    a minimum precision floor
    #
    # Threshold is now tuned on a clean, naturally-imbalanced validation
    # split (not on SMOTE-balanced data), so it should generalize to the
    # test set correctly instead of over-favoring Recall at Accuracy's
    # expense.
    run_xhealthguard_experiment(target_precision=None, threshold_objective='f1')
