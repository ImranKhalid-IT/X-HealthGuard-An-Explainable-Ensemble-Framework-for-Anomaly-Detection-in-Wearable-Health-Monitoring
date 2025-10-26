# X-HealthGuard: Advanced ECG Anomaly Detection

**X-HealthGuard** is an advanced ECG anomaly detection framework that leverages both machine learning and deep learning approaches. It combines HRV-based feature engineering, hyperparameter-tuned Random Forest classification, and optional 1D-CNN benchmarking to achieve high-performance detection on the MIT-BIH Arrhythmia Database.

## Features

**Advanced HRV Feature Engineering**
- Time-domain: mean RR, SDNN, RMSSD, pNN50
- Frequency-domain: VLF, LF, HF, LF/HF ratio
- Windowed ECG processing (30-second standard windows)

**Machine Learning Pipeline**
- Random Forest classifier with GridSearchCV hyperparameter tuning
- SMOTE for class balancing
- Performance metrics: Precision, Recall, F1-Score, Accuracy, AUC
- Confusion matrix visualization

**Deep Learning Benchmark (Optional)**
- 1D-CNN for ECG classification with convolutional, pooling, batch normalization, and dropout layers

**Automated Output**
- Generates performance tables
- Saves confusion matrix figures
- Supports reproducible experiments


## Installation

Clone the repository and install dependencies:

```bash
git clone <repository_url>
cd X-HealthGuard
pip install biosppy peakutils wfdb scipy scikit-learn pandas matplotlib seaborn tensorflow imbalanced-learn

Run
python paper_17_IEEE.py

Contact if you have any issues
Muhammad Imran Khalid
Email: <imran.khalid292@gmail.com>


