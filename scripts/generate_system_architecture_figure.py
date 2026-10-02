import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("Sepsis_Prediction_Model/figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis('off')

# Color Palette (Scientific Clinical AI)
c_input = "#e8f4f8"
c_feat = "#d1ecf1"
c_model1 = "#e2e3e5" # LogReg
c_model2 = "#d6e4f0" # RF
c_model3 = "#fce8e6" # XGB
c_model4 = "#e6f4ea" # LGBM
c_model5 = "#fef7e0" # DeepNet
c_meta = "#e8eaed"
c_out = "#feebe8"

edge_dark = "#2b5c8f"

def draw_box(ax, x, y, w, h, text, title=None, bg_color="#ffffff", border_color="#333333", title_color="#1a202c", font_size=10, title_size=11):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12,rounding_size=0.2",
        ec=border_color, fc=bg_color, lw=1.8, zorder=2
    )
    ax.add_patch(box)
    
    if title:
        ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', fontsize=title_size, fontweight='bold', color=title_color, zorder=3)
        ax.text(x + w/2, y + (h - 0.4)/2, text, ha='center', va='center', fontsize=font_size, color="#2d3748", zorder=3, multialignment='center')
    else:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=font_size, color="#2d3748", zorder=3, multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color="#4a5568", style="->", lw=2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3),
        zorder=1
    )

# 1. Top Input Layer
draw_box(ax, 3.5, 6.7, 8.0, 1.0, 
         "Longitudinal Hourly ICU Measurements (Vitals, Lab Tests, Demographics)\nDataset: PhysioNet Challenge 2019 (40,333 Admissions | 1.55M Hourly Records)", 
         title="RAW MULTIVARIATE CLINICAL DATA STREAM", 
         bg_color=c_input, border_color="#17a2b8", title_color="#0f6674")

# Arrow from Input to Feature Engineering
draw_arrow(ax, 7.5, 6.7, 7.5, 5.8)

# 2. Dynamic Feature Engineering Layer
draw_box(ax, 2.0, 4.8, 11.0, 1.0,
         "• 3h & 6h Moving Averages (Trend)   • 6h Rolling Volatility (std)   • 6h Extrema (min / max)\n• Bedside Hemodynamic Indices: Shock Index (HR / SBP), Pulse Pressure (SBP - DBP), qSOFA",
         title="DYNAMIC TEMPORAL FEATURE ENGINEERING & PHYSIOLOGICAL TRAJECTORIES",
         bg_color=c_feat, border_color="#0c5460", title_color="#0c5460")

# Arrows branching down to the 5 base paradigms
branch_xs = [1.5, 4.5, 7.5, 10.5, 13.5]
for bx in branch_xs:
    draw_arrow(ax, 7.5, 4.8, bx, 3.8)

# 3. Five Heterogeneous Base Models
base_models = [
    {"x": 0.3, "w": 2.4, "title": "Linear Anchor", "desc": "L2 Logistic Regression\nClass-Weighted\nLinear Baseline", "bg": c_model1, "border": "#6c757d"},
    {"x": 3.3, "w": 2.4, "title": "Bagging Ensemble", "desc": "Random Forest\n100 Bagged Trees\nVariance Minimization", "bg": c_model2, "border": "#0056b3"},
    {"x": 6.3, "w": 2.4, "title": "Depth-wise GBDT", "desc": "Histogram XGBoost\nDepth-wise Expansion\nscale_pos_weight=54.5", "bg": c_model3, "border": "#b02a37"},
    {"x": 9.3, "w": 2.4, "title": "Leaf-wise GBDT", "desc": "LightGBM\nLeaf-wise Splitting\nHigh-Order Interaction", "bg": c_model4, "border": "#198754"},
    {"x": 12.3, "w": 2.4, "title": "Deep Neural Net", "desc": "PyTorch DeepNet\nBatchNorm + Dropout\nRepresentation Learning", "bg": c_model5, "border": "#ffc107"}
]

for bm in base_models:
    draw_box(ax, bm["x"], 2.4, bm["w"], 1.4, bm["desc"], title=bm["title"], bg_color=bm["bg"], border_color=bm["border"], font_size=8.5, title_size=9.5)

# Arrows converging to Meta Learner
for bx in branch_xs:
    draw_arrow(ax, bx, 2.4, 7.5, 1.8)

# 4. Stacking Meta-Learner (Super Learner)
draw_box(ax, 3.0, 1.1, 9.0, 0.75,
         "Out-of-Fold Probability Blending: Meta Weights [LGBM: 1.80 | XGB: 1.67 | DeepNet: 0.90 | RF: 0.51 | LR: 0.52]\nPlatt Sigmoid Calibration (Brier Loss: 0.1258 → 0.0169)",
         title="HETEROGENEOUS STACKING META-LEARNER (SUPER LEARNER)",
         bg_color="#e2e8f0", border_color="#2b5c8f", title_color="#1a365d")

# Arrow to Final Outputs
draw_arrow(ax, 7.5, 1.1, 7.5, 0.75)

# 5. Bottom Clinical Decision Outputs
draw_box(ax, 2.5, 0.05, 10.0, 0.7,
         "AUROC: 0.8060 | Official PhysioNet 2019 Utility: 0.3435 | Median Early Warning: 29.5 Hours (74.5% Detection Rate)\nReal-Time Bedside Sepsis Alerting & Game-Theoretic SHAP Interpretability",
         title=None, bg_color=c_out, border_color="#c53030", font_size=9.5)

plt.tight_layout()
plt.savefig("figures/system_architecture.png", dpi=300, bbox_inches='tight')
plt.savefig("Sepsis_Prediction_Model/figures/system_architecture.png", dpi=300, bbox_inches='tight')
plt.close()
print("Generated high-resolution system_architecture.png successfully!")
