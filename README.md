Here is the complete README file in English, formatted and ready for your GitHub repository.

---

# X-HealthGuard: An Explainable Framework for Anomaly Detection in Ambulatory ECG

This repository contains the Python implementation of **X-HealthGuard**, an explainable AI (XAI) framework for binary Normal-versus-Anomaly anomaly detection in ambulatory ECG monitoring. The codebase strictly follows the methodology proposed in the research paper.

##  Project Overview

X-HealthGuard is a hybrid framework designed to balance high predictive performance with computational efficiency and interpretability. This repository implements two primary models:

1. **X-HealthGuard (Random Forest):** The explainable branch that processes 30-second ECG segments, extracts 256-dimensional (16x16) Heart Rate Variability (HRV) features, and generates classification outputs alongside counterfactual explanations.


2. **1D-CNN Benchmark:** An accuracy-oriented deep learning benchmark that processes 2-second (720 samples) raw ECG waveforms directly.



## ⚙️ How the Code Works

### 1. Preprocessing

* **Band-pass Filter:** A zero-phase 4th-order Butterworth band-pass filter (0.5 Hz to 40 Hz) is applied to remove noise and baseline wander.


* **R-Peak Detection:** R-peaks are detected using a Pan-Tompkins-based approach via the `biosppy` library.



### 2. HRV Feature Engineering

* Each 30-second ECG segment is decomposed into 16 overlapping 15-second local windows (with a 1-second stride).


* From each window, 16 distinct feature families (Time-domain, Frequency-domain, and Non-linear) are extracted.


* These are concatenated to form a single **256-dimensional feature vector (16x16)** per 30-second segment.



### 3. Model Training & Data Splitting

* **Group Hold-out:** To prevent data leakage, an 80:20 record-level grouping strategy is enforced so that segments from the same patient/record never overlap between training and testing partitions.


* **SMOTE Balancing:** Synthetic Minority Over-sampling Technique (SMOTE) is applied *exclusively* to the active training fold to handle class imbalance, leaving the test set completely untouched.


* **Random Forest:** The classifier is initialized with the paper's optimized hyperparameters (`n_estimators=300`, `max_depth=30`, `class_weight='balanced'`).



### 4. 1D-CNN (Deep Learning Benchmark)

* Implements the exact architecture from the paper (Table III): 3 Convolutional Blocks, MaxPooling1D layers, and a 256-unit Dense layer utilizing the `Adam` optimizer.



##  Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install numpy pandas wfdb scipy scikit-learn matplotlib seaborn biosppy imbalanced-learn tensorflow

```

### Execution

1. Clone this repository to your local machine.
2. Run the main execution script in your terminal:

```bash
python paper_17_ieee.py

```

Note: On the first run, the `wfdb` package will automatically download the required records from the MIT-BIH Arrhythmia Database and store them in a local `mitdb_data` directory.

##  Expected Outputs

Upon running the script, you will observe the following in your console:

1. **Feature Extraction:** Logs indicating the downloading of the database (if needed) and the progress of HRV feature extraction across the segments.


2. **Model Evaluation:** The script will output the primary binary classification metrics (Precision, Recall, F1-Score, and Accuracy) for the trained Random Forest model.


3. **Visualizations (Optional):** If confusion matrix plotting is enabled in the code, the resulting figures will be saved to your working directory.
