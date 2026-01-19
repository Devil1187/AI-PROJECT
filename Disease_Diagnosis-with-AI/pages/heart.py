import streamlit as st
import pickle
import warnings
import pandas as pd
from utils.ui_helpers import set_disease_background, disease_sidebar, mainmenu

# Page config
st.set_page_config(page_title="Heart Disease", page_icon="❤️", layout="wide")
warnings.filterwarnings("ignore")

# UI setup
set_disease_background("Image/Heart.png")
disease_sidebar("Heart")
mainmenu()

st.title("❤️ Heart Disease Prediction")

# Load the model
with open("Models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names based on your pipeline
categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
binary_features = ['sex', 'fbs', 'exang']

# Categorical mappings (raw input)
cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
restecg_options = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
slope_options = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
thal_options = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2
}
sex_options = {"Female": 0, "Male": 1}
fbs_options = {"No": 0, "Yes": 1}
exang_options = {"No": 0, "Yes": 1}

# UI Inputs
cols = st.columns(3)

with cols[0]:
    age = st.number_input("Age", min_value=1, max_value=120)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=50, max_value=600)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250)
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1)

with cols[1]:
    sex = st.selectbox("Sex", options=list(sex_options.keys()))
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=list(fbs_options.keys()))
    exang = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()))
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=list(slope_options.keys()))
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", min_value=0, max_value=3)

with cols[2]:
    cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()))
    restecg = st.selectbox("Resting Electrocardiographic Results", options=list(restecg_options.keys()))
    thal = st.selectbox("Thalassemia", options=list(thal_options.keys()))

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_options[sex]],
    'cp': [cp_options[cp]],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_options[fbs]],
    'restecg': [restecg_options[restecg]],
    'thalach': [thalach],
    'exang': [exang_options[exang]],
    'oldpeak': [oldpeak],
    'slope': [slope_options[slope]],
    'ca': [ca],
    'thal': [thal_options[thal]]
})

# Predict button
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)[0]
        st.session_state.update({
            'prediction_result': prediction,
            'selected_disease': "Heart Disease",
            'input_data': input_data.to_dict()
        })
        st.switch_page("pages/result.py")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Back button
if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
