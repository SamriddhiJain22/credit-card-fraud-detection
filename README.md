# credit-card-fraud-detection
"ML project to detect credit card fraud using Python &amp; scikit-learn."

# Credit Card Fraud Detection ML Project 🚀

## Overview
Built a pipeline to detect fraudulent transactions in imbalanced data (99.8% legit). Used Python, scikit-learn, XGBoost. Tuned model hits AUC 0.98, F1 0.92—catches 95%+ fraud!

![Dashboard Preview](fraud_dashboard.html) <!-- Embed or link -->

## Key Steps
1. **EDA:** Plots showed fraud in high amounts/late hours; V3/V7 key features.
2. **Prep:** Scaled, SMOTE balanced classes.
3. **Models:** XGBoost > Logistic; tuned with GridSearch.
4. **Eval:** Confusion matrix, ROC—low misses.
5. **Demo:** Predicts samples (e.g., $500 odd txn: 0.85 prob fraud).

## Results Table
| Model          | AUC   | F1    |
|----------------|-------|-------|
| Logistic      | 0.95  | 0.85  |
| XGBoost Tuned | 0.98  | 0.92  |

## Files
- `Fraud_Detection_Project.ipynb`: Full notebook.
- `xgb_fraud_model.pkl`: Deploy-ready model.
- `fraud_dashboard.html`: Interactive viz.

## Run It
1. `jupyter notebook Fraud_Detection_Project.ipynb`
2. For predictions: Load pickle & scale input.

## Future: Deploy as Streamlit App
```python
import streamlit as st
import pickle
# Load model...
txn_amount = st.number_input("Amount")
# Predict & show prob
