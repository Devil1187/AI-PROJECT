import base64
import streamlit as st

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def mainmenu():
    st.markdown("""
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def set_disease_background(image):
    with open( image , "rb") as f:
        bg = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container, [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(255, 255, 255, 0.4);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    [data-testid="stSidebar"] {{
        background-color: transparent !important;
        border-right: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def disease_sidebar(disease_type):
    """Disease-specific sidebar content"""
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=60)
    st.sidebar.markdown(f"### {disease_type} Health Tips")
    
    if disease_type == "Diabetes":
        st.sidebar.markdown("""
        **✅ Do's:**
        - Monitor blood sugar regularly
        - Eat high-fiber foods
        - Stay hydrated
        
        **❌ Don'ts:**
        - Avoid sugary drinks
        - Don't skip meals
        - Limit processed carbs
        """)
    elif disease_type == "Heart":
        st.sidebar.markdown("""
        **✅ Do's:**
        - Exercise 30 mins daily
        - Eat omega-3 rich foods
        - Manage stress
        
        **❌ Don'ts:**
        - Avoid trans fats
        - Don't smoke
        - Limit alcohol
        """)
    elif disease_type == "Fatty Liver":
        st.sidebar.markdown("""
        **✅ Do's:**
        - Lose weight gradually
        - Eat antioxidant-rich foods
        - Exercise regularly
        
        **❌ Don'ts:**
        - Avoid alcohol
        - Limit sugary foods
        - Don't eat late at night
        """)