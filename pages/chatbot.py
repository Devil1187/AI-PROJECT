import os
import sys
import types
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
## from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
from utils.prompt_templates import MEDICAL_ADVICE_PROMPT
from utils.ui_helpers import mainmenu, set_disease_background

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="AI Health Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Health Assistant")
set_disease_background("Image/chatbot.png")
mainmenu()
load_dotenv()

# --------------------------------------------------
# Patch torch.classes (KEEPING YOUR LOGIC)
# --------------------------------------------------
class TorchClassesStub(types.ModuleType):
    def __getattr__(self, name):
        raise AttributeError(f"torch.classes has no attribute '{name}'")

sys.modules["torch.classes"] = TorchClassesStub("torch.classes")

# --------------------------------------------------
# Environment safety
# --------------------------------------------------
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# --------------------------------------------------
# Load model ONCE (critical fix)
# --------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading AI model (first time only)...")
def load_llm():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=256,        # 🔥 reduced (faster)
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        device=device,
    )

    return HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# Response generator (robust handling)
# --------------------------------------------------
def generate_advice(patient_data, llm):
    prompt = PromptTemplate.from_template(MEDICAL_ADVICE_PROMPT)
    user_prompt = prompt.format(**patient_data)

    # 🔥 TinyLlama REQUIRES chat-style formatting
    formatted_prompt = f"""
<|system|>
You are a professional medical assistant. Give clear, safe lifestyle advice.
</s>
<|user|>
{user_prompt}
</s>
<|assistant|>
"""

    response = llm.invoke(formatted_prompt)

    # Make response safe
    if not isinstance(response, str):
        response = str(response)

    response = response.strip()

    # 🔴 FALLBACK (important)
    if len(response) < 10:
        return "⚠️ I couldn't generate advice right now. Please try again."

    return response

# --------------------------------------------------
# Main UI
# --------------------------------------------------
def main():
    st.subheader("Get personalized health advice based on your profile")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous assistant messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
        st.header("Patient Profile")

        with st.form("patient_form"):
            disease = st.selectbox(
                "Known Health Condition",
                ["-- Select --", "Diabetes", "Heart Disease", "Fatty Liver", "Fit & Fine"]
            )
            alcohol_status = st.selectbox(
                "Alcohol Consumption",
                ["-- Select --", "Alcoholic", "Non-Alcoholic"]
            )
            junk_food_frequency = st.selectbox(
                "Junk Food Consumption",
                ["-- Select --", "Regular", "Few days in a week", "No"]
            )
            physical_activity_level = st.selectbox(
                "Physical Activity",
                ["-- Select --", "Regular", "Few days in a week", "No"]
            )
            area_type = st.selectbox(
                "Living Area",
                ["-- Select --", "Urban", "Rural"]
            )
            pollution_level = st.selectbox(
                "Pollution Level in Area",
                ["-- Select --", "High", "Medium", "Low"]
            )

            submitted = st.form_submit_button("Get Health Advice")

    # 🔥 Load model OUTSIDE submit block
    llm = load_llm()
    st.success("✅ AI model ready")

    if submitted:
        if disease == "-- Select --":
            st.error("Please select a health condition")
            return

        patient_data = {
            "disease": disease,
            "alcohol_status": alcohol_status,
            "junk_food_frequency": junk_food_frequency,
            "physical_activity_level": physical_activity_level,
            "mental_wellness_activity": "Not specified",
            "area_type": area_type,
            "pollution_level": pollution_level
        }

        st.session_state.messages.append(
            {"role": "user", "content": "Patient profile submitted"}
        )

        with st.spinner("🧠 Generating health advice..."):
            advice = generate_advice(patient_data, llm)

            with st.chat_message("assistant"):
                st.markdown(advice)

            st.session_state.messages.append(
                {"role": "assistant", "content": advice}
            )

# --------------------------------------------------
# Navigation
# --------------------------------------------------
if st.button("🏠 Back to Home"):
    st.switch_page("app.py")

main()
