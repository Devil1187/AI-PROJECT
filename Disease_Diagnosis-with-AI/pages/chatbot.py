import os
from dotenv import load_dotenv
import streamlit as st
import torch

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

from utils.prompt_templates import MEDICAL_ADVICE_PROMPT
from utils.ui_helpers import mainmenu, set_disease_background

st.set_page_config(page_title="AI Health Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Health Assistant")
set_disease_background("Image/chatbot.png")
mainmenu()
load_dotenv()

# Patch torch.classes before anything loads it


# Set Streamlit + CUDA environment variables
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Cache the model loading for performance
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        device=-1,
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def generate_advice(patient_data, llm):
    prompt = PromptTemplate.from_template(MEDICAL_ADVICE_PROMPT)
    formatted_prompt = prompt.format(**patient_data)

    response = llm.invoke(formatted_prompt)

    if isinstance(response, str):
        # Strip the prompt content if it appears in the generated response
        response = response.replace(formatted_prompt.strip(), "").strip()

        # Optional: Also remove the first few lines if it still looks like instructions
        response_lines = response.split("\n")
        clean_lines = [line for line in response_lines if not line.lower().startswith("you are") and line.strip() != ""]
        response = "\n".join(clean_lines).strip()

    return response

def main():
    st.title("Medical Lifestyle Advisor")
    st.subheader("Get personalized health advice based on your profile")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Only display assistant messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
    
    with st.sidebar:
        st.sidebar.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
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
    
    llm = load_llm()
    
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

        # User input is not displayed, but could be stored if needed
        st.session_state.messages.append({"role": "user", "content": "Patient profile submitted."})

        with st.spinner("Generating health advice..."):
            try:
                advice = generate_advice(patient_data, llm)
                with st.chat_message("assistant"):
                    st.markdown(advice)
                st.session_state.messages.append({"role": "assistant", "content": advice})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if st.button("🏠 Back to Home"):
    st.switch_page("app.py")

main()
