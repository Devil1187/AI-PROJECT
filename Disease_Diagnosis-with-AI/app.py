import streamlit as st
from utils.ui_helpers import get_base64_of_bin_file

st.set_page_config(page_title="Disease Prediction Home", page_icon="⚕️", layout="centered")

# Inject fullscreen Spline background
spline_bg = """
<style>
body {
    margin: 0;
    overflow: hidden;
    object-fit: cover;
}
#spline-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 110%;
    height: 111%;
    z-index: -1;
    border: none;
}
.stApp {
    background: transparent;
}
</style>
<iframe id="spline-bg" src="https://my.spline.design/pillanddnaanimation-oHEADbdOIaEQNS23hH4hTNQZ/?hideUI=1" frameborder="0"></iframe>
"""
st.markdown(spline_bg, unsafe_allow_html=True)

# Custom styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    #MainMenu, header, footer {visibility: hidden;}

    .stButton>button {
        width: 100% !important;
        height: 3.5rem;
        font-size: 1.2rem;
        font-weight: bolder;
        padding: 0.75rem 1.25rem;
        margin: 0.75rem 0;
        text-align: center;
        border-radius: 12px;
        color: #000000;
        background-color: rgba(255, 255, 255, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(8px);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.35);
        color: #000000;
    }

    .block-container {
        padding-top: 5rem;
    }

    .viewer-frame {
        margin-top: 2rem;
        width: 100%;
        height: 600px;
        border: none;
        border-radius: 14px;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)


logo_base64 = get_base64_of_bin_file("Image/logo1.png")



st.markdown(
    f"""
    <style>
        .logo-container {{
            position: fixed;
            top: 10px;
            left: 10px; /* use 'right: 10px;' for top-right */
            z-index: 100;
        }}
        .logo-container img {{
            height: 60px; /* control the logo size */
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)


# Main content
st.title("🧠 Disease Prediction System")
st.subheader("Choose a prediction module")

col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button("🔬 Diabetes Prediction"):
        st.switch_page("pages/diabetes.py")
    if st.button("🫀 Heart Disease Prediction"):
        st.switch_page("pages/heart.py")
    if st.button("💬 Go to Chatbot"):
        st.switch_page("pages/chatbot.py")    

with col2:
    if st.button("🧬 Fatty Liver Prediction"):
        st.switch_page("pages/fattyliver.py")
    if st.button("📰 Health Blog"):
        st.switch_page("pages/blog.py")