import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

MODEL_PATH = Path("models/best_model.joblib")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = Path("data/diabetes.csv")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model = load_model()
metrics = load_metrics()

st.title("🩺 Diabetes Prediction")
st.caption(f"Model: **{metrics['best_model']}** · Trained on the Pima Indians Diabetes dataset")

tab_predict, tab_models, tab_data = st.tabs(["Predict", "Model comparison", "Data explorer"])

# ---------------- Predict ----------------
with tab_predict:
    st.subheader("Patient information")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 40, 250, 120)
        bp = st.number_input("Blood pressure (mm Hg)", 30, 130, 70)
        skin = st.number_input("Skin thickness (mm)", 5, 100, 20)
    with c2:
        insulin = st.number_input("Insulin (mu U/ml)", 10, 900, 80)
        bmi = st.number_input("BMI", 10.0, 70.0, 28.0, step=0.1)
        dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.40, step=0.01)
        age = st.number_input("Age", 15, 100, 30)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }])[metrics["features"]]

        proba = float(model.predict_proba(row)[0][1])
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Diabetes probability", f"{proba:.1%}")
        if proba < 0.33:
            m2.success("Low risk")
        elif proba < 0.66:
            m2.warning("Moderate risk")
        else:
            m2.error("High risk")
        st.progress(proba)

    st.info("⚠️ Educational project only — not a medical diagnosis. Consult a doctor.")

# ---------------- Model comparison ----------------
with tab_models:
    res = pd.DataFrame(metrics["results"]).set_index("Algorithm")
    st.dataframe(res.style.highlight_max(axis=0), use_container_width=True)
    st.bar_chart(res[["Accuracy", "Recall", "F1"]], stack=False)
    st.write(f"Confusion matrix of **{metrics['best_model']}** (rows = actual, columns = predicted):")
    st.dataframe(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"],
    ))

# ---------------- Data explorer ----------------
with tab_data:
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        st.write(f"{data.shape[0]} rows × {data.shape[1]} columns")
        st.dataframe(data.head(20), use_container_width=True)
        st.write("Outcome distribution")
        st.bar_chart(data["Outcome"].value_counts())
        st.write("Summary statistics")
        st.dataframe(data.describe(), use_container_width=True)
    else:
        st.warning("Put diabetes.csv inside the data/ folder to see this tab.")
