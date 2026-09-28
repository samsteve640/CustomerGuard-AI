import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CustomerGuard AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#f6f8fc;}
[data-testid="stSidebar"] {background:#0b1830;}
/* Sidebar background */
[data-testid="stSidebar"] {
    background: #0b1830;
}

/* Sidebar headings and normal text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #f8fafc !important;
}

/* Text inside input/select boxes */
[data-testid="stSidebar"] input {
    color: #111827 !important;
}

/* Selected value inside dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #111827 !important;
}

/* Dropdown options */
[data-baseweb="popover"] * {
    color: #111827 !important;
}
.block-container {padding-top:1.5rem;max-width:1250px;}
div[data-testid="stMetric"] {background:white;border:1px solid #e5eaf1;padding:16px;border-radius:16px;box-shadow:0 5px 16px rgba(15,23,42,.05);}
.stButton>button {border-radius:10px;font-weight:700;min-height:44px;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "CustomerGuard_AI_Logistic_Model.joblib"
DATA_PATH = "telco 2.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

DROP_COLUMNS = [
    "Customer ID","Customer Status","Churn Score","Churn Category","Churn Reason",
    "Satisfaction Score","Country","State","City","Zip Code","Latitude","Longitude","Quarter"
]
X = df.drop(columns=DROP_COLUMNS + ["Churn Label"], errors="ignore")

st.markdown("""
<div style="background:linear-gradient(120deg,#0b1830,#143b67 58%,#0f6b78);padding:30px 34px;border-radius:22px;color:white;margin-bottom:20px;box-shadow:0 10px 30px rgba(15,23,42,.12)">
<div style="font-size:.75rem;letter-spacing:.1em;font-weight:700;opacity:.8">MACHINE LEARNING • RETENTION INTELLIGENCE</div>
<h1 style="margin:8px 0">🛡️ CustomerGuard AI</h1>
<p style="margin:0;opacity:.88">Identify churn risk early, understand customer signals, and support smarter retention decisions.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Customer Profile")
    st.write("Adjust the customer details below, then analyse churn risk.")

    defaults = X.iloc[0].copy()
    user = {}

    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                user[col] = 0.0
                continue
            mn, mx, med = float(clean.min()), float(clean.max()), float(clean.median())
            if np.all(np.isclose(clean % 1, 0)) and mx < 10000:
                user[col] = st.number_input(col, min_value=int(mn), max_value=int(mx),
                                            value=int(round(float(defaults[col]))) if pd.notna(defaults[col]) else int(round(med)))
            else:
                user[col] = st.number_input(col, min_value=mn, max_value=mx,
                                            value=float(defaults[col]) if pd.notna(defaults[col]) else med)
        else:
            opts = [str(x) for x in series.dropna().unique().tolist()]
            default = str(defaults[col]) if pd.notna(defaults[col]) else (opts[0] if opts else "")
            idx = opts.index(default) if default in opts else 0
            user[col] = st.selectbox(col, opts, index=idx)

    analyse = st.button("Analyse Churn Risk", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Portfolio Analytics", "About the Model"])

with tab1:
    if analyse:
        input_df = pd.DataFrame([user])[X.columns]
        probability = float(model.predict_proba(input_df)[0, 1])

        if probability >= 0.70:
            risk = "HIGH"
        elif probability >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{probability:.1%}")
        c2.metric("Risk Level", risk)
        c3.metric("Model ROC-AUC", "90.9%")

        st.progress(min(max(probability, 0.0), 1.0))

        st.markdown("### Retention Recommendations")
        actions = []
        if user.get("Contract") == "Month-to-Month":
            actions.append("Offer an appropriate longer-term contract incentive.")
        if str(user.get("Premium Tech Support")) == "No":
            actions.append("Offer technical-support onboarding or a trial support package.")
        if "Monthly Charge" in user and float(user["Monthly Charge"]) > float(df["Monthly Charge"].median()):
            actions.append("Review plan value and consider a targeted retention offer.")
        if "Tenure in Months" in user and float(user["Tenure in Months"]) < 12:
            actions.append("Prioritise early-life engagement and proactive service check-ins.")
        if not actions:
            actions.append("Continue standard retention monitoring and customer engagement.")

        for a in actions[:3]:
            st.write("•", a)

        st.info("Predictions are decision-support signals, not proof that a customer will churn. Retention actions should be reviewed by a human.")
    else:
        st.info("Use the customer profile in the sidebar and click **Analyse Churn Risk**.")

with tab2:
    st.markdown("### Customer Churn Overview")
    total = len(df)
    churned = int((df["Churn Label"] == "Yes").sum())
    churn_rate = churned / total

    a,b,c = st.columns(3)
    a.metric("Customers", f"{total:,}")
    b.metric("Observed Churners", f"{churned:,}")
    c.metric("Observed Churn Rate", f"{churn_rate:.1%}")

    contract = pd.crosstab(df["Contract"], df["Churn Label"], normalize="index")
    if "Yes" in contract.columns:
        st.markdown("### Churn Rate by Contract")
        st.bar_chart((contract["Yes"] * 100).sort_values(ascending=False))

    st.markdown("### Monthly Charge by Churn Status")
    st.bar_chart(df.groupby("Churn Label")["Monthly Charge"].mean())

    st.markdown("### Average Tenure by Churn Status")
    st.bar_chart(df.groupby("Churn Label")["Tenure in Months"].mean())

with tab3:
    st.markdown("""
### Model
**Logistic Regression** with an end-to-end Scikit-learn preprocessing pipeline.

### Test-set performance
- **Accuracy:** 80.55%
- **Precision:** 58.99%
- **Recall:** 87.70%
- **F1-score:** 70.54%
- **ROC-AUC:** 90.91%

### Design rationale
The project prioritises recall because failing to identify a genuine churner can represent a missed retention opportunity. Variables that directly reveal or closely follow the churn outcome—such as Churn Reason, Churn Category, Churn Score and Customer Status—were excluded to reduce target leakage.

### Responsible use
CustomerGuard AI is a portfolio decision-support prototype. Predictive associations do not establish causation, and automated outputs should not replace human review.
""")

st.divider()
st.caption("CustomerGuard AI · Portfolio project by Stephen Samson · Data Science & Artificial Intelligence")
