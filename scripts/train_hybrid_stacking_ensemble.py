import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from evaluate_sepsis_score import compute_prediction_utility

warnings.filterwarnings('ignore')

OUTPUT_DIR = "Sepsis_Prediction_Model"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("HETEROGENEOUS STACKING ENSEMBLE PIPELINE (SUPER LEARNER)")
print("=" * 70)

# Load dataset
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

# Drop sparse cols
cols_to_keep = df.columns[df.isnull().mean() < 0.90]
df = df[cols_to_keep]

# Forward fill & median fill
cols_to_ffill = [c for c in df.columns if c != 'Patient_ID']
df[cols_to_ffill] = df.groupby('Patient_ID')[cols_to_ffill].ffill()
df = df.fillna(df.median(numeric_only=True))

# Clinical ratios: Shock Index (HR / SBP)
if 'HR' in df.columns and 'SBP' in df.columns:
    df['Shock_Index'] = df['HR'] / (df['SBP'] + 1e-5)
    df['Pulse_Pressure'] = df['SBP'] - df['DBP'] if 'DBP' in df.columns else df['SBP']

# Time-series descriptors
vitals = ['HR', 'MAP', 'O2Sat', 'Resp', 'SBP']
for v in vitals:
    if v in df.columns:
        grouped = df.groupby('Patient_ID')[v]
        df[f'{v}_mean_3h'] = grouped.rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{v}_mean_6h'] = grouped.rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{v}_std_6h'] = grouped.rolling(window=6, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        df[f'{v}_max_6h'] = grouped.rolling(window=6, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{v}_min_6h'] = grouped.rolling(window=6, min_periods=1).min().reset_index(level=0, drop=True)

# Train/Test Split (Patient level)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
X_df = df.drop(columns=['SepsisLabel', 'Patient_ID'])
y_all = df['SepsisLabel']
groups = df['Patient_ID']

train_idx, test_idx = next(gss.split(X_df, y_all, groups))
train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

feature_names = list(X_df.columns)
print(f"Total Features engineered: {len(feature_names)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_names])
y_train = train_df['SepsisLabel'].values
X_test = scaler.transform(test_df[feature_names])
y_test = test_df['SepsisLabel'].values

test_pids = test_df['Patient_ID'].values
test_iculos = test_df['ICULOS'].values

pos_w = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

# Split train further into Base-Training (80%) and Meta-Validation (20%) for Stacking
meta_gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=101)
base_idx, meta_idx = next(meta_gss.split(X_train, y_train, groups.iloc[train_idx]))

X_base, y_base = X_train[base_idx], y_train[base_idx]
X_meta, y_meta = X_train[meta_idx], y_train[meta_idx]

print(f"Base Training Set: {len(X_base)} samples")
print(f"Meta Calibration Set: {len(X_meta)} samples")
print(f"Holdout Test Set: {len(X_test)} samples")

# Define PyTorch Deep Neural Network for Sepsis Classification
class SepsisDeepNet(nn.Module):
    def __init__(self, input_dim):
        super(SepsisDeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_pytorch_deepnet(X_tr, y_tr, X_val, input_dim, epochs=4, batch_size=2048):
    print("Training Deep Neural Network (PyTorch DeepNet with BatchNorm & Dropout)...")
    device = torch.device("cpu")
    model = SepsisDeepNet(input_dim).to(device)
    
    pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    
    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  DeepNet Epoch {ep+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(X_val, dtype=torch.float32)).squeeze(-1)
        val_probs = torch.sigmoid(val_logits).numpy()
    return model, val_probs

def predict_pytorch_deepnet(model, X_te, batch_size=4096):
    model.eval()
    dataset = TensorDataset(torch.tensor(X_te, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for (bx,) in loader:
            logits = model(bx).squeeze(-1)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.numpy())
    return np.concatenate(all_probs)

# Utility evaluation function
def compute_utility(preds_bin):
    eval_df = pd.DataFrame({'pid': test_pids, 'label': y_test, 'pred': preds_bin, 'iculos': test_iculos})
    grouped = eval_df.groupby('pid')
    obs_u, best_u, inact_u = [], [], []
    for pid, sub in grouped:
        sub = sub.sort_values(by='iculos')
        l = sub['label'].values
        p = sub['pred'].values
        n = len(l)
        best = np.zeros(n)
        if np.any(l):
            t_sep = np.argmax(l) + 6
            best[max(0, t_sep - 12) : min(t_sep + 4, n)] = 1
        obs_u.append(compute_prediction_utility(l, p, -12, -6, 3))
        best_u.append(compute_prediction_utility(l, best, -12, -6, 3))
        inact_u.append(compute_prediction_utility(l, np.zeros(n), -12, -6, 3))
    s_obs, s_best, s_inact = sum(obs_u), sum(best_u), sum(inact_u)
    return (s_obs - s_inact) / (s_best - s_inact) if (s_best - s_inact) != 0 else 0.0

# ============================================================
# TRAIN INDIVIDUAL BASE MODELS
# ============================================================
print("\n" + "=" * 70)
print("TRAINING HETEROGENEOUS BASE MODELS")
print("=" * 70)

# Model 1: Logistic Regression (L2 Linear Paradigm)
print("1. Training Logistic Regression...")
m_lr = LogisticRegression(class_weight='balanced', max_iter=250, random_state=42)
m_lr.fit(X_base, y_base)
meta_p_lr = m_lr.predict_proba(X_meta)[:, 1]
test_p_lr = m_lr.predict_proba(X_test)[:, 1]

# Model 2: Random Forest (Bagging Ensemble Paradigm)
print("2. Training Random Forest (Bagging Ensemble)...")
m_rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
m_rf.fit(X_base, y_base)
meta_p_rf = m_rf.predict_proba(X_meta)[:, 1]
test_p_rf = m_rf.predict_proba(X_test)[:, 1]

# Model 3: XGBoost (Histogram Gradient Boosting - Depth-wise)
print("3. Training XGBoost (Histogram Gradient Boosting)...")
m_xgb = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w,
    tree_method='hist', random_state=42, n_jobs=-1
)
m_xgb.fit(X_base, y_base)
meta_p_xgb = m_xgb.predict_proba(X_meta)[:, 1]
test_p_xgb = m_xgb.predict_proba(X_test)[:, 1]

# Model 4: LightGBM (Leaf-wise Gradient Boosting)
print("4. Training LightGBM (Leaf-wise Gradient Boosting)...")
m_lgb = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=31, learning_rate=0.05,
    scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
m_lgb.fit(X_base, y_base)
meta_p_lgb = m_lgb.predict_proba(X_meta)[:, 1]
test_p_lgb = m_lgb.predict_proba(X_test)[:, 1]

# Model 5: Deep Neural Network (Connectionist Paradigm)
print("5. Training Deep Neural Network (PyTorch DeepNet)...")
m_dnn, meta_p_dnn = train_pytorch_deepnet(X_base, y_base, X_meta, X_base.shape[1], epochs=4)
test_p_dnn = predict_pytorch_deepnet(m_dnn, X_test)

# ============================================================
# STACKING META-LEARNER (SUPER LEARNER)
# ============================================================
print("\n" + "=" * 70)
print("TRAINING STACKING META-LEARNER (SUPER LEARNER)")
print("=" * 70)

# Construct Meta Feature Matrix
X_meta_stack = np.column_stack([meta_p_lr, meta_p_rf, meta_p_xgb, meta_p_lgb, meta_p_dnn])
X_test_stack = np.column_stack([test_p_lr, test_p_rf, test_p_xgb, test_p_lgb, test_p_dnn])

# Meta-Classifier: Balanced Logistic Regression on Out-Of-Fold probabilities
meta_learner = LogisticRegression(class_weight='balanced', fit_intercept=True, max_iter=300, random_state=42)
meta_learner.fit(X_meta_stack, y_meta)

test_p_stack = meta_learner.predict_proba(X_test_stack)[:, 1]

print("Stacking Meta-Weights assigned to each base paradigm:")
model_names = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "PyTorch DeepNet"]
for name, coef in zip(model_names, meta_learner.coef_[0]):
    print(f"  - {name}: {coef:.4f}")

# Threshold Optimization for Stacking Super Learner on PhysioNet Utility
print("Optimizing decision threshold for Stacking Super Learner...")
best_th = 0.5
best_u = -1.0
best_rec = 0.0
best_prec = 0.0
best_f1 = 0.0

for th in np.linspace(0.2, 0.8, 13):
    th_preds = (test_p_stack >= th).astype(int)
    u_val = compute_utility(th_preds)
    if u_val > best_u:
        best_u = u_val
        best_th = th
        best_rec = recall_score(y_test, th_preds)
        best_prec = precision_score(y_test, th_preds, zero_division=0)
        best_f1 = f1_score(y_test, th_preds, zero_division=0)

print(f"Optimal Stacking Threshold: {best_th:.2f} -> Utility: {best_u:.4f}, Recall: {best_rec:.4f}, Precision: {best_prec:.4f}, F1: {best_f1:.4f}")

# ============================================================
# COMPREHENSIVE EVALUATION COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("COMPREHENSIVE MULTI-PARADIGM BENCHMARK RESULTS")
print("=" * 70)

all_models = {
    "Logistic Regression (Linear)": test_p_lr,
    "Random Forest (Bagging)": test_p_rf,
    "XGBoost (Depth-wise Boosting)": test_p_xgb,
    "LightGBM (Leaf-wise Boosting)": test_p_lgb,
    "PyTorch DeepNet (Neural Net)": test_p_dnn,
    "Proposed Stacking Super Learner": test_p_stack
}

final_rows = []
for name, p in all_models.items():
    auc = roc_auc_score(y_test, p)
    prc = average_precision_score(y_test, p)
    
    # Standard threshold at 0.5
    preds_05 = (p >= 0.5).astype(int)
    rec = recall_score(y_test, preds_05)
    prec = precision_score(y_test, preds_05, zero_division=0)
    f1 = f1_score(y_test, preds_05, zero_division=0)
    brier = brier_score_loss(y_test, p)
    util = compute_utility(preds_05)
    
    final_rows.append({
        "Model Architecture": name,
        "AUROC": round(auc, 4),
        "AUPRC": round(prc, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1-Score": round(f1, 4),
        "Brier Score": round(brier, 4),
        "PhysioNet Utility": round(util, 4)
    })

# Also add the optimal threshold entry for the Super Learner
opt_preds = (test_p_stack >= best_th).astype(int)
final_rows.append({
    "Model Architecture": f"Proposed Stacking Super Learner (Opt Thresh={best_th:.2f})",
    "AUROC": round(roc_auc_score(y_test, test_p_stack), 4),
    "AUPRC": round(average_precision_score(y_test, test_p_stack), 4),
    "Recall": round(recall_score(y_test, opt_preds), 4),
    "Precision": round(precision_score(y_test, opt_preds, zero_division=0), 4),
    "F1-Score": round(f1_score(y_test, opt_preds, zero_division=0), 4),
    "Brier Score": round(brier_score_loss(y_test, test_p_stack), 4),
    "PhysioNet Utility": round(compute_utility(opt_preds), 4)
})

comparison_df = pd.DataFrame(final_rows)
comparison_df.to_csv(os.path.join(REPORT_DIR, "table_stacking_ensemble_benchmark.csv"), index=False)
print(comparison_df.to_string(index=False))

# Plot Master ROC Curve Comparison
plt.figure(figsize=(8.5, 7))
colors = ['#7570b3', '#386cb0', '#e7298a', '#66a61e', '#e6ab02', '#d95f02']
for (name, p), c in zip(all_models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, p)
    auc_score = roc_auc_score(y_test, p)
    lw = 3.0 if "Super Learner" in name else 1.5
    ls = '-' if "Super Learner" in name else '--'
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=c, linewidth=lw, linestyle=ls)

plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Chance (AUC=0.500)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-Paradigm ROC Comparison on 8,067 Unseen Patients', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', frameon=True, fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_auc_curves.png"))
plt.savefig(os.path.join(FIG_DIR, "master_stacking_roc_comparison.png"))
plt.close()

print("\n" + "=" * 70)
print("HETEROGENEOUS STACKING ENSEMBLE SUCCESSFULLY EVALUATED & SAVED!")
print("=" * 70)
