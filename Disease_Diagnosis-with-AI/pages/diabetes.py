import streamlit as st
import pickle
import warnings
from utils.ui_helpers import set_disease_background, disease_sidebar, mainmenu

warnings.filterwarnings("ignore")



st.set_page_config(page_title="Diabetes Prediction", page_icon="🔬", layout="wide")
set_disease_background("Image/Diabetes.png")
disease_sidebar("Diabetes")  # Disease-specific sidebar
mainmenu()


st.title("🔬 Diabetes Prediction")

with open("Models/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature labels and placeholders (excluding pregnancy for now)
features = [
    "Glucose Level", "Blood Pressure value", "Skin Thickness value",
    "Insulin Level", "BMI value", "Diabetes Pedigree Function", "Age"
]

placeholder = [
    "52.2 - 274 mg/dL", "5.9 - 115 mm Hg", "2.9 - 23.3 mm",
    "50 - 180 μU/mL ", "2.61 - 41.62 kg/m²", "0.08 - 2.42", "Enter your age"
]

inputs = []
cols = st.columns(3)

# 1. Gender select box in first column
gender = cols[0].selectbox("Select Gender", [ "Male", "Female"])

# 2. Pregnancy input always takes a column, but is conditionally active
if gender == "Female":
    preg_input = cols[1].text_input("Number of Pregnancies", placeholder="No. pregnancies")
else:
    preg_input = cols[1].text_input("Number of Pregnancies", value="0", disabled=True )
inputs.append(preg_input)

# 3. Remaining inputs align from next available column
for i, feat in enumerate(features):
    col = cols[(i + 2) % 3]  # Shift by 2 to account for gender and pregnancy fields
    value = col.text_input(feat, placeholder=placeholder[i])
    inputs.append(value)

# Predict button
if st.button("🔍 Predict"):
    if "" in inputs :
        st.error("Please select a gender and fill in all the fields before submitting.")
    else:
        try:
            data = [float(i) for i in inputs]
            st.session_state.update({
                'prediction_result': model.predict([data])[0],
                'selected_disease': "Diabetes",
                'input_data': data
            })
            st.switch_page("pages/result.py")
        except ValueError:
            st.error("Please enter valid numeric values only.")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
