import streamlit as st
import pandas as pd
import joblib

from utils.ui_helpers import set_disease_background, disease_sidebar, mainmenu

st.set_page_config(page_title="Fatty Liver", page_icon="🧬", layout="wide")
set_disease_background("Image/Liver.png")
disease_sidebar("Fatty Liver")
mainmenu()

st.title("🧬 Fatty Liver Risk Prediction")

# Load model
model = joblib.load("Models/fatty_liver_model.joblib")

# Inputs
age = st.number_input("Age", 18, 90)
gender = st.selectbox("Gender", ["Female", "Male"])
weight = st.number_input("Weight (kg)", 30.0, 200.0)
height = st.number_input("Height (cm)", 120.0, 220.0)

male = 1 if gender == "Male" else 0
bmi = weight / ((height / 100) ** 2)

if bmi < 10 or bmi > 60:
    st.warning("BMI value looks medically unrealistic.")

if st.button("🔍 Predict Risk"):
    input_df = pd.DataFrame([{
        "age": age,
        "male": male,
        "bmi": bmi
    }])

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("🧠 Prediction Result")
    st.write(f"**Fatty Liver Risk Probability:** `{prob*100:.2f}%`")

    if prob > 0.7:
        st.error("⚠️ High Risk – medical consultation recommended")
    elif prob > 0.4:
        st.warning("⚠️ Moderate Risk – lifestyle changes advised")
    else:
        st.success("✅ Low Risk")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
