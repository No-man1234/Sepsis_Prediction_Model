# A Real-Time Early Warning System for ICU Sepsis Prediction
## Using a Heterogeneous Stacking Ensemble (Super Learner)

**Authors:** Abdullah Al Noman, Md. Zawad Al Mahir, Md. Rakibul Islam, Mahathir Mohammad  
*Department of Computer Science and Engineering, United International University, Dhaka, Bangladesh*  
*Corresponding Email:* `md.zawadalmahir12554@gmail.com`

---

## 1. Project Overview
This repository contains the publication-grade codebase, experimental benchmarks, and clinical models for an advanced **ICU Sepsis Early Warning System**. Using the full PhysioNet/Computing in Cardiology Challenge 2019 dataset ($1,552,113$ hourly observations across $40,333$ ICU admissions), our framework combines dynamic temporal feature engineering with a **Heterogeneous Stacking Super-Learner** to predict sepsis onset well in advance of clinical shock.

### Key Breakthroughs:
* **AUROC:** **`0.8060`** on an independent holdout test cohort of $8,067$ unseen patients ($308,785$ hourly records).
* **Official PhysioNet 2019 Clinical Utility Score ($U_{norm}$):** **`0.3435`** (matches top international challenge benchmarks).
* **Probability Calibration:** Reduced Brier loss from $0.1258$ to **`0.0169`** using Platt scaling.
* **Clinical Lead Time:** Detected **74.5%** of sepsis cases early or on-time with a **median lead time of 29.5 hours** (mean 50.3 hours) before overt clinical diagnosis.
* **Multi-Paradigm Stacking:** Fuses 5 distinct model paradigms:
  1. *Linear Discriminative:* $L_2$ Logistic Regression
  2. *Bootstrap Bagging:* Random Forest (100 Trees)
  3. *Depth-wise Gradient Boosting:* Histogram XGBoost
  4. *Leaf-wise Gradient Boosting:* LightGBM
  5. *Connectionist Deep Learning:* Custom PyTorch DeepNet with BatchNorm & Dropout

---

## 2. Directory Structure

```
DM-Project/
│
├── Sepsis_SuperLearner_Publication_Colab.ipynb # Primary Colab Jupyter notebook
├── README.md                            # Project documentation & guide
├── requirements.txt                     # Project dependencies
│
├── figures/                             # 300 DPI Publication-Grade Figures
│   ├── system_architecture.png          # System architecture diagram (Figure 1)
│   ├── master_stacking_roc_comparison.png # Multi-model ROC curves
│   ├── precision_recall_curve.png       # Precision-Recall curves
│   ├── ablation_study_chart.png         # Feature engineering ablation bar chart
│   ├── calibration_curve.png            # Platt calibration reliability diagram
│   ├── decision_curve_analysis.png      # Net clinical benefit curves (DCA)
│   ├── early_warning_distribution.png   # Lead time histogram & KDE distribution
│   ├── patient_simulation_early_warning.png # Real-time patient monitoring trajectory
│   ├── shap_summary.png                 # Game-theoretic SHAP feature impact
│   ├── confusion_matrices.png           # Multi-model confusion matrices
│   ├── utility_vs_threshold.png         # Clinical utility vs threshold curves
│   └── vitals_distribution.png          # Physiological parameter distributions
│
├── notebooks/                           # Jupyter & Colab Notebooks
│   ├── Sepsis_SuperLearner_Publication_Colab.ipynb # Primary publication notebook
│   └── Sepsis_Detection_Colab_legacy.ipynb        # Archived early baseline version
│
├── models/                              # Trained Model Checkpoints & Scalers
│   ├── xgb_model.pkl                    # Base XGBoost classifier
│   ├── xgboost_model_advanced.pkl       # Hyper-tuned XGBoost model
│   └── rf_model.pkl                     # Random Forest baseline
│
├── reports/                             # Generated Scientific Evaluation Tables
│   ├── table_stacking_ensemble_benchmark.csv # Multi-paradigm tournament table
│   ├── table_ablation_study.csv              # 4-stage ablation study metrics
│   ├── table_cross_validation_results.csv    # 5-fold patient cross-validation
│   └── table_model_benchmarks.csv            # Baseline comparisons
│
└── scripts/                             # Core Python Experiment Scripts
    ├── train_hybrid_stacking_ensemble.py     # End-to-end Stacking Super Learner pipeline
    ├── run_publication_experiments.py        # Benchmark, ablation, and DCA runner
    ├── evaluate_sepsis_score.py              # Official PhysioNet 2019 scoring utility
    ├── generate_system_architecture_figure.py# System diagram generator
    └── download_physionet.py                 # PhysioNet 2019 dataset downloader
```

---

## 3. Quick Start & Execution

### Environment Setup
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the End-to-End Experiment Pipeline
1. **Download Data:**
   ```bash
   python scripts/download_physionet.py
   ```
2. **Train the Stacking Super Learner:**
   ```bash
   python scripts/train_hybrid_stacking_ensemble.py
   ```
3. **Execute Full Benchmarking & Clinical Utility Evaluation:**
   ```bash
   python scripts/run_publication_experiments.py
   ```
4. **Generate System Architecture Diagram:**
   ```bash
   python scripts/generate_system_architecture_figure.py
   ```

---

## 4. How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Sepsis_SuperLearner_Publication_Colab.ipynb` (or `notebooks/Sepsis_SuperLearner_Publication_Colab.ipynb`).
3. In Cell 1, mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Place `data/processed_sepsis_data_FULL.csv` in your Google Drive under `/content/drive/MyDrive/Sepsis_Detection_Project/`.
5. Select **Runtime -> Run all**. The notebook will train all 5 models, optimize the stacking meta-learner, and export all figures and tables directly to your Drive.

---

## 5. Benchmark Summary

| Model Paradigm | AUROC | AUPRC | Recall | Precision | F1-Score | Brier Loss | PhysioNet Utility ($U_{norm}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **qSOFA Baseline ($\ge 2$)** | 0.5784 | 0.0223 | 0.0696 | 0.0352 | 0.0468 | 0.1216 | 0.0160 |
| **Logistic Regression ($L_2$)** | 0.7317 | 0.0706 | 0.6082 | 0.0419 | 0.0784 | 0.1988 | 0.2103 |
| **Random Forest (100 Trees)** | 0.7786 | 0.0791 | 0.3944 | 0.0915 | 0.1485 | 0.1137 | 0.2875 |
| **PyTorch DeepNet** | 0.7871 | 0.0901 | **0.6983** | 0.0443 | 0.0833 | 0.1874 | 0.2655 |
| **Histogram XGBoost** | 0.7940 | 0.0979 | 0.5634 | 0.0646 | 0.1158 | 0.1231 | 0.3312 |
| **LightGBM** | 0.7995 | 0.0936 | 0.6067 | 0.0593 | 0.1080 | 0.1328 | 0.3322 |
| **Proposed Stacking Super Learner** | **0.8060** | **0.0985** | 0.5692 | **0.0665** | **0.1192** | **0.0169** | **0.3435** |
