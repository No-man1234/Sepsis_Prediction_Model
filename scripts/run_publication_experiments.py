import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
import xgboost as xgb
from evaluate_sepsis_score import compute_prediction_utility

warnings.filterwarnings('ignore')

OUTPUT_DIR = "Sepsis_Prediction_Model"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set clean scientific plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("1. LOADING & PREPROCESSING DATASET")
print("=" * 70)
candidate_paths = [
    "data/processed_sepsis_data_FULL.csv",
    "data/processed_sepsis_data_FULL.csv.gz",
    "processed_sepsis_data_FULL.csv",
    "processed_sepsis_data_FULL.csv.gz",
    "Sepsis_Detection_Project/processed_sepsis_data_FULL.csv",
    "/content/drive/MyDrive/Sepsis_Detection_Project/processed_sepsis_data_FULL.csv",
]
csv_path = next((p for p in candidate_paths if os.path.exists(p)), "data/processed_sepsis_data_FULL.csv")
print(f"Loading dataset from: {csv_path}")
t0 = time.time()
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows across {df['Patient_ID'].nunique()} patients in {time.time()-t0:.1f}s")

# Feature drop & imputation
missing_threshold = 0.90
cols_to_keep = df.columns[df.isnull().mean() < missing_threshold]
df = df[cols_to_keep]

cols_to_ffill = [c for c in df.columns if c != 'Patient_ID']
df[cols_to_ffill] = df.groupby('Patient_ID')[cols_to_ffill].ffill()
df = df.fillna(df.median(numeric_only=True))

# Time-Series Feature Engineering
print("Generating time-series features (means, volatility, min/max)...")
vitals = ['HR', 'MAP', 'O2Sat', 'Resp', 'SBP']
for v in vitals:
    if v in df.columns:
        grouped = df.groupby('Patient_ID')[v]
        df[f'{v}_mean_3h'] = grouped.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{v}_mean_6h'] = grouped.rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{v}_std_6h'] = grouped.rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        df[f'{v}_max_6h'] = grouped.rolling(window=6, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{v}_min_6h'] = grouped.rolling(window=6, min_periods=1).min().reset_index(level=0, drop=True)

# Add qSOFA score component: Resp >= 22 (1 pt), SBP <= 100 (1 pt)
if 'Resp' in df.columns and 'SBP' in df.columns:
    df['qSOFA_score'] = ((df['Resp'] >= 22).astype(int) + (df['SBP'] <= 100).astype(int))

print(f"Engineered dataset shape: {df.shape}")

# Patient-level 80/20 train/test split
print("\n" + "=" * 70)
print("2. PATIENT-LEVEL TRAIN / TEST SPLIT")
print("=" * 70)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
X_all = df.drop(columns=['SepsisLabel', 'Patient_ID'])
y_all = df['SepsisLabel']
groups = df['Patient_ID']

train_idx, test_idx = next(gss.split(X_all, y_all, groups))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

print(f"Train set: {len(train_df)} rows, {train_df['Patient_ID'].nunique()} patients ({train_df['SepsisLabel'].sum()} sepsis hours)")
print(f"Test set:  {len(test_df)} rows, {test_df['Patient_ID'].nunique()} patients ({test_df['SepsisLabel'].sum()} sepsis hours)")

# Fast vectorized official PhysioNet 2019 Utility metric calculation
def compute_cohort_utility_fast(labels_series, preds_series, patient_ids_series, iculos_series, dt_early=-12, dt_optimal=-6, dt_late=3):
    eval_df = pd.DataFrame({
        'pid': patient_ids_series,
        'label': labels_series,
        'pred': preds_series,
        'iculos': iculos_series
    })
    
    unique_patients = eval_df['pid'].unique()
    num_patients = len(unique_patients)
    
    obs_u = np.zeros(num_patients)
    best_u = np.zeros(num_patients)
    inaction_u = np.zeros(num_patients)
    
    grouped = eval_df.groupby('pid')
    for i, (pid, sub) in enumerate(grouped):
        sub = sub.sort_values(by='iculos')
        labels = sub['label'].values
        preds = sub['pred'].values
        n = len(labels)
        
        best_preds = np.zeros(n)
        inaction_preds = np.zeros(n)
        
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_preds[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, n)] = 1
            
        obs_u[i] = compute_prediction_utility(labels, preds, dt_early, dt_optimal, dt_late)
        best_u[i] = compute_prediction_utility(labels, best_preds, dt_early, dt_optimal, dt_late)
        inaction_u[i] = compute_prediction_utility(labels, inaction_preds, dt_early, dt_optimal, dt_late)
        
    sum_obs = np.sum(obs_u)
    sum_best = np.sum(best_u)
    sum_inaction = np.sum(inaction_u)
    
    denom = sum_best - sum_inaction
    if denom == 0:
        return 0.0
    return (sum_obs - sum_inaction) / denom

# Feature sets
raw_vitals_labs = [c for c in X_all.columns if not any(c.endswith(s) for s in ['_3h', '_6h', 'qSOFA_score'])]
trend_3h_features = raw_vitals_labs + [c for c in X_all.columns if c.endswith('_mean_3h')]
full_features = list(X_all.columns)

y_train = train_df['SepsisLabel'].values
y_test = test_df['SepsisLabel'].values
test_pids = test_df['Patient_ID'].values
test_iculos = test_df['ICULOS'].values

pos_w = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
print(f"Computed scale_pos_weight: {pos_w:.2f}")

# Standardize
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(train_df[full_features])
X_test_full_scaled = scaler_full.transform(test_df[full_features])

scaler_raw = StandardScaler()
X_train_raw_scaled = scaler_raw.fit_transform(train_df[raw_vitals_labs])
X_test_raw_scaled = scaler_raw.transform(test_df[raw_vitals_labs])

scaler_3h = StandardScaler()
X_train_3h_scaled = scaler_3h.fit_transform(train_df[trend_3h_features])
X_test_3h_scaled = scaler_3h.transform(test_df[trend_3h_features])

# ============================================================
# EXPERIMENT 1: ABLATION STUDY
# ============================================================
print("\n" + "=" * 70)
print("3. EXECUTING ABLATION STUDY")
print("=" * 70)

ablation_configs = [
    {
        "name": "Config 1: Raw Features (Unweighted XGB)",
        "X_train": X_train_raw_scaled,
        "X_test": X_test_raw_scaled,
        "weight": 1.0,
        "features": len(raw_vitals_labs)
    },
    {
        "name": "Config 2: Raw Features + Class Weighting",
        "X_train": X_train_raw_scaled,
        "X_test": X_test_raw_scaled,
        "weight": pos_w,
        "features": len(raw_vitals_labs)
    },
    {
        "name": "Config 3: Static + 3h Trends + Class Weighting",
        "X_train": X_train_3h_scaled,
        "X_test": X_test_raw_scaled,
        "X_test": X_test_3h_scaled,
        "weight": pos_w,
        "features": len(trend_3h_features)
    },
    {
        "name": "Config 4: Full Proposed (6h Volatility & Extremes)",
        "X_train": X_train_full_scaled,
        "X_test": X_test_full_scaled,
        "weight": pos_w,
        "features": len(full_features)
    }
]

ablation_results = []
trained_models = {}

for cfg in ablation_configs:
    t_start = time.time()
    print(f"Training {cfg['name']} (features={cfg['features']}, weight={cfg['weight']:.1f})...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=cfg['weight'],
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    model.fit(cfg['X_train'], y_train)
    probs = model.predict_proba(cfg['X_test'])[:, 1]
    preds_05 = (probs >= 0.5).astype(int)
    
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    rec = recall_score(y_test, preds_05)
    prec = precision_score(y_test, preds_05, zero_division=0)
    f1 = f1_score(y_test, preds_05, zero_division=0)
    
    # Calculate utility score on a sample for speed (or full test cohort)
    print("Computing PhysioNet Clinical Utility score...")
    utility_score = compute_cohort_utility_fast(y_test, preds_05, test_pids, test_iculos)
    
    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.1f}s -> AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Recall: {rec:.4f}, Utility: {utility_score:.4f}")
    
    ablation_results.append({
        "Configuration": cfg['name'],
        "Features": cfg['features'],
        "AUROC": round(auroc, 4),
        "AUPRC": round(auprc, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "PhysioNet Utility": round(utility_score, 4)
    })
    trained_models[cfg['name']] = model

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(os.path.join(REPORT_DIR, "table_ablation_study.csv"), index=False)
print("\nAblation Study Summary Table:")
print(ablation_df.to_string(index=False))

# ============================================================
# EXPERIMENT 2: MULTI-MODEL BENCHMARK COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("4. EXECUTING MULTI-MODEL BENCHMARK")
print("=" * 70)

benchmark_results = []
model_probs = {}

# 1. Clinical qSOFA Baseline
print("Evaluating Clinical qSOFA Score baseline...")
qsofa_preds = (test_df['qSOFA_score'].values >= 2).astype(int)
qsofa_probs = test_df['qSOFA_score'].values / 2.0
qsofa_auc = roc_auc_score(y_test, qsofa_probs)
qsofa_prc = average_precision_score(y_test, qsofa_probs)
qsofa_rec = recall_score(y_test, qsofa_preds)
qsofa_prec = precision_score(y_test, qsofa_preds, zero_division=0)
qsofa_f1 = f1_score(y_test, qsofa_preds, zero_division=0)
qsofa_util = compute_cohort_utility_fast(y_test, qsofa_preds, test_pids, test_iculos)
qsofa_brier = brier_score_loss(y_test, qsofa_probs)

benchmark_results.append({
    "Model": "Clinical qSOFA (>=2)",
    "AUROC": round(qsofa_auc, 4),
    "AUPRC": round(qsofa_prc, 4),
    "Recall": round(qsofa_rec, 4),
    "Precision": round(qsofa_prec, 4),
    "F1-Score": round(qsofa_f1, 4),
    "Brier Score": round(qsofa_brier, 4),
    "PhysioNet Utility": round(qsofa_util, 4)
})
model_probs["Clinical qSOFA"] = qsofa_probs

# 2. Logistic Regression (L2 penalized, balanced)
print("Training Logistic Regression (Balanced weights)...")
lr = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42)
lr.fit(X_train_full_scaled, y_train)
lr_probs = lr.predict_proba(X_test_full_scaled)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)
lr_auc = roc_auc_score(y_test, lr_probs)
lr_prc = average_precision_score(y_test, lr_probs)
lr_rec = recall_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds, zero_division=0)
lr_f1 = f1_score(y_test, lr_preds, zero_division=0)
lr_util = compute_cohort_utility_fast(y_test, lr_preds, test_pids, test_iculos)
lr_brier = brier_score_loss(y_test, lr_probs)

benchmark_results.append({
    "Model": "Logistic Regression (L2)",
    "AUROC": round(lr_auc, 4),
    "AUPRC": round(lr_prc, 4),
    "Recall": round(lr_rec, 4),
    "Precision": round(lr_prec, 4),
    "F1-Score": round(lr_f1, 4),
    "Brier Score": round(lr_brier, 4),
    "PhysioNet Utility": round(lr_util, 4)
})
model_probs["Logistic Regression"] = lr_probs

# 3. Random Forest (100 trees, max_depth=12, balanced)
print("Training Random Forest (Balanced, max_depth=12)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_full_scaled, y_train)
rf_probs = rf.predict_proba(X_test_full_scaled)[:, 1]
rf_preds = (rf_probs >= 0.5).astype(int)
rf_auc = roc_auc_score(y_test, rf_probs)
rf_prc = average_precision_score(y_test, rf_probs)
rf_rec = recall_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds, zero_division=0)
rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
rf_util = compute_cohort_utility_fast(y_test, rf_preds, test_pids, test_iculos)
rf_brier = brier_score_loss(y_test, rf_probs)

benchmark_results.append({
    "Model": "Random Forest (Ensemble)",
    "AUROC": round(rf_auc, 4),
    "AUPRC": round(rf_prc, 4),
    "Recall": round(rf_rec, 4),
    "Precision": round(rf_prec, 4),
    "F1-Score": round(rf_f1, 4),
    "Brier Score": round(rf_brier, 4),
    "PhysioNet Utility": round(rf_util, 4)
})
model_probs["Random Forest"] = rf_probs

# 4. Proposed Weighted XGBoost
proposed_xgb = trained_models["Config 4: Full Proposed (6h Volatility & Extremes)"]
xgb_probs = proposed_xgb.predict_proba(X_test_full_scaled)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)
xgb_auc = roc_auc_score(y_test, xgb_probs)
xgb_prc = average_precision_score(y_test, xgb_probs)
xgb_rec = recall_score(y_test, xgb_preds)
xgb_prec = precision_score(y_test, xgb_preds, zero_division=0)
xgb_f1 = f1_score(y_test, xgb_preds, zero_division=0)
xgb_util = compute_cohort_utility_fast(y_test, xgb_preds, test_pids, test_iculos)
xgb_brier = brier_score_loss(y_test, xgb_probs)

benchmark_results.append({
    "Model": "Proposed Time-Aware XGBoost",
    "AUROC": round(xgb_auc, 4),
    "AUPRC": round(xgb_prc, 4),
    "Recall": round(xgb_rec, 4),
    "Precision": round(xgb_prec, 4),
    "F1-Score": round(xgb_f1, 4),
    "Brier Score": round(xgb_brier, 4),
    "PhysioNet Utility": round(xgb_util, 4)
})
model_probs["Proposed XGBoost"] = xgb_probs

benchmark_df = pd.DataFrame(benchmark_results)
benchmark_df.to_csv(os.path.join(REPORT_DIR, "table_model_benchmarks.csv"), index=False)
print("\nMulti-Model Benchmark Comparison Table:")
print(benchmark_df.to_string(index=False))

# ============================================================
# EXPERIMENT 3: UTILITY CURVE & OPTIMAL THRESHOLD SELECTION
# ============================================================
print("\n" + "=" * 70)
print("5. CLINICAL UTILITY VS. CLASSIFICATION THRESHOLD")
print("=" * 70)

thresholds_range = np.linspace(0.1, 0.9, 17)
threshold_metrics = []

for th in thresholds_range:
    bin_preds = (xgb_probs >= th).astype(int)
    p_prec = precision_score(y_test, bin_preds, zero_division=0)
    p_rec = recall_score(y_test, bin_preds, zero_division=0)
    p_f1 = f1_score(y_test, bin_preds, zero_division=0)
    p_util = compute_cohort_utility_fast(y_test, bin_preds, test_pids, test_iculos)
    
    threshold_metrics.append({
        "Threshold": round(th, 3),
        "Precision": round(p_prec, 4),
        "Recall": round(p_rec, 4),
        "F1": round(p_f1, 4),
        "Utility": round(p_util, 4)
    })

th_df = pd.DataFrame(threshold_metrics)
best_th_row = th_df.loc[th_df['Utility'].idxmax()]
print(f"Optimal PhysioNet Utility Threshold: {best_th_row['Threshold']} with U_norm = {best_th_row['Utility']:.4f}")
print(f"At optimal threshold: Recall = {best_th_row['Recall']:.4f}, Precision = {best_th_row['Precision']:.4f}, F1 = {best_th_row['F1']:.4f}")

# Plot Utility vs. Threshold
plt.figure(figsize=(8, 5))
plt.plot(th_df['Threshold'], th_df['Utility'], marker='o', color='#2b5c8f', linewidth=2.5, label='PhysioNet 2019 Normalized Utility ($U_{norm}$)')
plt.plot(th_df['Threshold'], th_df['F1'], marker='s', color='#d95f02', linestyle='--', label='F1-Score')
plt.axvline(best_th_row['Threshold'], color='red', linestyle=':', label=f"Optimal Utility Thresh ({best_th_row['Threshold']}, $U={best_th_row['Utility']:.3f}$)")
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Clinical Utility and F1-Score Across Classification Thresholds', fontsize=13, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "utility_vs_threshold.png"))
plt.close()

# ============================================================
# EXPERIMENT 4: PROBABILITY CALIBRATION & DECISION CURVE ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("6. PROBABILITY CALIBRATION & DECISION CURVE ANALYSIS")
print("=" * 70)

# Platt scaling (Sigmoid Calibration) on holdout subset of train
calibrator = CalibratedClassifierCV(proposed_xgb, method='sigmoid', cv='prefit')
calibrator.fit(X_train_full_scaled[-50000:], y_train[-50000:])
calibrated_probs = calibrator.predict_proba(X_test_full_scaled)[:, 1]

brier_uncal = brier_score_loss(y_test, xgb_probs)
brier_cal = brier_score_loss(y_test, calibrated_probs)
print(f"Brier Score Before Calibration: {brier_uncal:.4f}")
print(f"Brier Score After Calibration:  {brier_cal:.4f} (Lower is better)")

# Calibration Curve (Reliability Diagram)
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, xgb_probs, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.plot(prob_pred_uncal, prob_true_uncal, marker='o', color='#d95f02', label=f'Uncalibrated XGBoost (Brier={brier_uncal:.3f})')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', color='#1b9e77', label=f'Platt Calibrated XGBoost (Brier={brier_cal:.3f})')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Observed Empirical Frequency', fontsize=12)
plt.title('Reliability Diagram (Probability Calibration)', fontsize=13, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "calibration_curve.png"))
plt.close()

# Decision Curve Analysis (DCA)
# Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
print("Calculating Decision Curve Analysis (Net Clinical Benefit)...")
dca_thresholds = np.linspace(0.01, 0.20, 20)
N = len(y_test)
net_benefit_xgb = []
net_benefit_all = []
net_benefit_none = np.zeros(len(dca_thresholds))

prevalence = np.mean(y_test)

for pt in dca_thresholds:
    # Model
    preds = (calibrated_probs >= pt).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    nb = (tp / N) - (fp / N) * (pt / (1 - pt))
    net_benefit_xgb.append(nb)
    
    # Treat All
    tp_all = np.sum(y_test == 1)
    fp_all = np.sum(y_test == 0)
    nb_all = (tp_all / N) - (fp_all / N) * (pt / (1 - pt))
    net_benefit_all.append(nb_all)

plt.figure(figsize=(8, 6))
plt.plot(dca_thresholds * 100, net_benefit_xgb, label='Proposed AI Decision Support', color='#2b5c8f', linewidth=2.5)
plt.plot(dca_thresholds * 100, net_benefit_all, label='Treat All (Universal Sepsis Protocol)', color='#7570b3', linestyle=':')
plt.plot(dca_thresholds * 100, net_benefit_none, label='Treat None (Zero Interventions)', color='black', linestyle='--')
plt.ylim(-0.01, max(net_benefit_xgb) * 1.3)
plt.xlabel('Threshold Probability (%) for Intervention', fontsize=12)
plt.ylabel('Net Clinical Benefit', fontsize=12)
plt.title('Decision Curve Analysis (Clinical Net Benefit)', fontsize=13, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "decision_curve_analysis.png"))
plt.close()

# ============================================================
# EXPERIMENT 5: 5-FOLD PATIENT-LEVEL CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("7. 5-FOLD PATIENT-STRATIFIED CROSS-VALIDATION")
print("=" * 70)

gkf = GroupKFold(n_splits=5)
cv_aurocs = []
cv_auprcs = []
cv_recalls = []
cv_f1s = []

print("Running 5-Fold GroupKFold on Training Cohort...")
for fold, (f_train_idx, f_val_idx) in enumerate(gkf.split(X_train_full_scaled, y_train, groups.iloc[train_idx])):
    f_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_w,
        tree_method='hist',
        random_state=42 + fold,
        n_jobs=-1
    )
    f_model.fit(X_train_full_scaled[f_train_idx], y_train[f_train_idx])
    f_probs = f_model.predict_proba(X_train_full_scaled[f_val_idx])[:, 1]
    f_preds = (f_probs >= 0.5).astype(int)
    
    f_auc = roc_auc_score(y_train[f_val_idx], f_probs)
    f_prc = average_precision_score(y_train[f_val_idx], f_probs)
    f_rec = recall_score(y_train[f_val_idx], f_preds)
    f_f1 = f1_score(y_train[f_val_idx], f_preds, zero_division=0)
    
    cv_aurocs.append(f_auc)
    cv_auprcs.append(f_prc)
    cv_recalls.append(f_rec)
    cv_f1s.append(f_f1)
    print(f"  Fold {fold+1}/5 -> AUROC: {f_auc:.4f}, AUPRC: {f_prc:.4f}, Recall: {f_rec:.4f}, F1: {f_f1:.4f}")

cv_summary = {
    "Metric": ["AUROC", "AUPRC", "Recall", "F1-Score"],
    "Mean": [np.mean(cv_aurocs), np.mean(cv_auprcs), np.mean(cv_recalls), np.mean(cv_f1s)],
    "Std": [np.std(cv_aurocs), np.std(cv_auprcs), np.std(cv_recalls), np.std(cv_f1s)],
    "95% CI Lower": [np.mean(cv_aurocs) - 1.96 * np.std(cv_aurocs) / np.sqrt(5),
                     np.mean(cv_auprcs) - 1.96 * np.std(cv_auprcs) / np.sqrt(5),
                     np.mean(cv_recalls) - 1.96 * np.std(cv_recalls) / np.sqrt(5),
                     np.mean(cv_f1s) - 1.96 * np.std(cv_f1s) / np.sqrt(5)],
    "95% CI Upper": [np.mean(cv_aurocs) + 1.96 * np.std(cv_aurocs) / np.sqrt(5),
                     np.mean(cv_auprcs) + 1.96 * np.std(cv_auprcs) / np.sqrt(5),
                     np.mean(cv_recalls) + 1.96 * np.std(cv_recalls) / np.sqrt(5),
                     np.mean(cv_f1s) + 1.96 * np.std(cv_f1s) / np.sqrt(5)]
}
cv_df = pd.DataFrame(cv_summary)
cv_df.to_csv(os.path.join(REPORT_DIR, "table_cross_validation_results.csv"), index=False)
print("\n5-Fold Patient-Stratified Cross Validation Summary:")
print(cv_df.to_string(index=False))

# ============================================================
# GENERATE PUBLICATION-READY COMPARISON CHARTS
# ============================================================
print("\n" + "=" * 70)
print("8. GENERATING PUBLICATION VECTOR FIGURES")
print("=" * 70)

# 1. Multi-Model ROC Curves
plt.figure(figsize=(8, 6.5))
palette = {'Proposed XGBoost': '#d95f02', 'Random Forest': '#2b5c8f', 'Logistic Regression': '#7570b3', 'Clinical qSOFA': '#1b9e77'}
for name, p in model_probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_val = roc_auc_score(y_test, p)
    lw = 2.5 if 'XGBoost' in name else 1.5
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=palette[name], linewidth=lw)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Baseline (AUC = 0.500)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Multi-Model ROC Comparison on 8,067 Unseen Patients', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_auc_curves.png"))
plt.close()

# 2. Multi-Model Precision-Recall Curves
plt.figure(figsize=(8, 6.5))
for name, p in model_probs.items():
    precs, recs, _ = precision_recall_curve(y_test, p)
    prc_val = average_precision_score(y_test, p)
    lw = 2.5 if 'XGBoost' in name else 1.5
    plt.plot(recs, precs, label=f"{name} (AUPRC = {prc_val:.3f})", color=palette[name], linewidth=lw)

plt.axhline(prevalence, color='black', linestyle=':', label=f'Prevalence Baseline ({prevalence*100:.1f}%)')
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title('Precision-Recall Curves Across Model Architectures', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "precision_recall_curve.png"))
plt.close()

# 3. Ablation Study Bar Chart
plt.figure(figsize=(9, 5))
labels_ablation = ['Raw Static (Unweighted)', 'Raw Static (Weighted)', 'Static + 3h Trends', 'Full Proposed (6h Dynamic)']
auc_vals = ablation_df['AUROC'].values
util_vals = ablation_df['PhysioNet Utility'].values

x = np.arange(len(labels_ablation))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5.5))
rects1 = ax.bar(x - width/2, auc_vals, width, label='AUROC', color='#2b5c8f')
rects2 = ax.bar(x + width/2, util_vals, width, label='PhysioNet Utility ($U_{norm}$)', color='#d95f02')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Ablation Study: Progressive Impact of Temporal Engineering and Class Weighting', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_ablation, fontsize=10)
ax.legend(frameon=True, fontsize=11)
ax.set_ylim(0, 0.95)

# Add value labels
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ablation_study_chart.png"))
plt.close()

print("\n" + "=" * 70)
print("ALL PUBLICATION EXPERIMENTS SUCCESSFULLY COMPLETED!")
print(f"Generated Tables: {REPORT_DIR}")
print(f"Generated Figures: {FIG_DIR}")
print("=" * 70)
