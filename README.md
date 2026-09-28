# CustomerGuard AI

An end-to-end customer churn prediction and retention intelligence portfolio project.

## What it does
CustomerGuard AI uses a Scikit-learn Logistic Regression pipeline to estimate customer churn probability, classify risk as Low/Medium/High, and surface practical retention actions.

## Model performance
- Accuracy: 80.55%
- Precision: 58.99%
- Recall: 87.70%
- F1-score: 70.54%
- ROC-AUC: 90.91%

## Run locally
1. Put `app.py`, `telco 2.csv`, and `CustomerGuard_AI_Logistic_Model.joblib` in the same folder.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run:
   `streamlit run app.py`

## Portfolio talking points
- Leakage-aware feature selection
- Mixed numeric/categorical preprocessing with Scikit-learn pipelines
- Class-imbalance-aware Logistic Regression
- Business-oriented model evaluation emphasizing recall and ROC-AUC
- Customer-level risk segmentation and retention recommendations
- Human-in-the-loop decision support
