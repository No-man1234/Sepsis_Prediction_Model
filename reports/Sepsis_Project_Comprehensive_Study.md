# Building a Real-Time Early Warning System for ICU Sepsis Prediction

## 1. Introduction and First Approach

This Data Mining project set out to address sepsis, one of the deadliest and most costly conditions in hospitals worldwide. We used the PhysioNet 2019 Challenge dataset, which holds more than 40,000 ICU patients and millions of hourly records. The goal was to predict sepsis onset before a doctor makes the clinical diagnosis.

**The Starting Point:**
We began with a standard data science pipeline. We loaded the raw hourly data, handled missing values with forward-filling and median imputation, and split the data into training and testing sets. Sepsis is rare: only about 2% of all recorded ICU hours are positive for it. To balance the dataset, we used **SMOTE (Synthetic Minority Over-sampling Technique)**. We then trained a baseline Random Forest classifier.

This first model gave reasonable baseline numbers, but we found critical flaws in the method when we looked closer.

## 2. Finding Flaws and Changing Direction

When we reviewed the first pipeline, we found two major problems that forced us to change our approach:

1. **The Time-Series SMOTE Trap:** We saw that applying SMOTE to chronological, hour-by-hour physiological data is problematic. SMOTE creates synthetic patients by blending data points. In a time-series context, this breaks the natural progression of a patient's vital signs and causes data leakage.
2. **Static vs. Dynamic Physiology:** The first model examined one hour at a time, in isolation. But human physiology is dynamic. A heart rate of 100 BPM may be normal for an active patient. A spike from 60 to 100 BPM over two hours is a warning sign. Our model could not see these trends.

**The Pivot:** We removed the SMOTE method entirely. Instead of treating the data as static snapshots, we moved to a time-series forecasting approach.

## 3. The New Approach: Feature Engineering and Weighted XGBoost

To detect patient deterioration, we added **Advanced Time-Series Feature Engineering**. For each critical vital sign (Heart Rate, O2 Saturation, Systolic Blood Pressure, Respiration, and others), we built rolling 6-hour windows. From these windows we calculated:

- **Volatility (`_std_6h`):** How much does the patient's heart rate change over six hours?
- **Extremes (`_max_6h`, `_min_6h`):** What was the worst blood pressure drop in the last shift?

To handle the 98% to 2% class imbalance without SMOTE, we switched to a **Hyper-Tuned XGBoost Classifier**. We used its built-in `scale_pos_weight` parameter. This setting forces the algorithm to penalize itself for missing a sepsis case. The model learns from real, unaltered patient data and balances the classes on its own.

## 4. Analysis of Final Results

The new architecture produced strong results:

- **AUROC (0.795):** The model achieved an AUROC of nearly 0.80. It can distinguish between septic and non-septic patients with high accuracy.
- **Recall Improvement:** The model found 3,053 true positive sepsis cases. This 55% recall rate more than doubled our first baseline.
- **Interpretability (SHAP):** We used SHAP values to interpret the model. The analysis showed that the model relied on our engineered features, especially `HR_max_6h` and `MAP_min_6h`. Medically, a rising heart rate combined with falling mean arterial pressure is the definition of Septic Shock. The model learned the same clinical patterns that doctors use.

## 5. Clinical Impact: The "Early Warning" Metric

To measure the clinical value of this system, we ran a simulation on all 577 unseen sepsis patients in the test set. We calculated when the model would trigger an alarm compared to the actual clinical diagnosis.

**The Results:**

- **Detection Rate:** The model caught 74.5% of sepsis cases (430 patients) early or on time.
- **Median Early Warning Time:** 29.5 hours before diagnosis.
- **Average Early Warning Time:** 50.3 hours before diagnosis.

**What this means:** In a real ICU, this system would alert nurses and doctors to a deteriorating patient more than a full day before clinical symptoms appear. Each hour of delayed antibiotics increases sepsis mortality by 8%. A 29.5-hour head start gives doctors a large window to intervene, give fluids and antibiotics, and save lives.

## 6. Future Improvements

To take this project from a successful academic study to a medical tool that can be deployed in hospitals, we would make these improvements:

1. **External Validation:** Test the model on a separate dataset, such as the eICU Collaborative Research Database or AmsterdamUMCdb. This would make sure the learned patterns generalize to different hospitals and patient groups.
2. **Deep Learning (LSTMs / Transformers):** Our rolling 6-hour windows captured short-term trends well. An LSTM network or Time-Series Transformer could process a patient's full 3-week ICU stay and find subtle, long-term patterns of decline.
3. **Integration with EHR Systems:** The final step is to build an API that connects this model directly into a hospital's Electronic Health Record (EHR) system, such as Epic or Cerner. This would give doctors a live, updating risk dashboard.
