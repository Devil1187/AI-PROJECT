import streamlit as st
import pickle
from utils.ui_helpers import set_disease_background, disease_sidebar, mainmenu
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fatty Liver", page_icon="🧬", layout="wide")
set_disease_background("Image/Liver.png")
disease_sidebar("Fatty Liver")
mainmenu()

st.title("🧬 Fatty Liver Prediction")
st.title("Under Development")

with open("Models/fatty_liver.pkl", "rb") as f:
    model = pickle.load(f)

features = ["age", "gender", "weight", "height"]
inputs = []
cols = st.columns(3)

for i, feat in enumerate(features):
    with cols[i % 3]:
        if feat == "gender":
            gender = st.selectbox("Gender", ["Female", "Male"])
            gender_value = 0 if gender == "Female" else 1
            inputs.append(gender_value)
        else:
            # Set placeholders
            placeholder_text = ""
            if feat == "age":
                placeholder_text = "YYYY"
            elif feat == "weight":
                placeholder_text = "Kilo-Grams"
            elif feat == "height":
                placeholder_text = "Centi-Meter"
                
            value = st.text_input(feat.capitalize(), placeholder=placeholder_text)
            inputs.append(value)

if st.button("🔍 Predict"):
    try:
        # Ensure all non-gender inputs are non-empty
        non_gender_inputs = [inputs[idx] for idx in range(len(features)) if features[idx] != "gender"]
        if any(val.strip() == "" for val in non_gender_inputs):
            st.error("Please fill in all the fields before submitting.")
        else:
            final_data = []
            for idx, val in enumerate(inputs):
                if features[idx] == "gender":
                    final_data.append(int(val))  # already numeric
                elif features[idx] == "age":
                    final_data.append(int(val))  # convert age to int
                else:
                    final_data.append(float(val))  # convert weight and height to float

            # Perform prediction
            st.session_state.update({
                'prediction_result': model.predict([final_data])[0],
                'selected_disease': "Fatty Liver",
                'input_data': final_data
            })
            st.switch_page("pages/result.py")
    except ValueError as e:
        st.error("Please enter valid numeric values only.")
        st.warning(f"Details: {e}")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
