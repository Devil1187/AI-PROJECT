import streamlit as st
from utils.ui_helpers import mainmenu , set_disease_background

st.set_page_config(page_title="Prediction Result", page_icon="📋", layout="wide")
st.title("🧾 Prediction Result")
mainmenu()
set_disease_background("Image/Result.png")

if "prediction_result" not in st.session_state:
    st.warning("No prediction data found.")
    st.stop()

result = st.session_state['prediction_result']
selected = st.session_state['selected_disease']
data = st.session_state['input_data']

st.markdown(f"### 🩺 Prediction for: **{selected}**")
st.markdown("#### Entered Parameters:")
for i, val in enumerate(data):
    st.write(f"- {i+1}. **{val}**")

st.markdown("---")
st.text(f"{result}")
if result == 1:
    st.error(f"🔴 Likely **{selected} Positive**.")
else:
    st.success(f"🟢 Likely **Not {selected} Positive**.")

if st.button("💬 Go to Chatbot"):
    st.switch_page("pages/chatbot.py")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
