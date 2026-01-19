import streamlit as st
from utils.ui_helpers import mainmenu , set_disease_background


st.set_page_config(page_title="Health Blog", page_icon="📰", layout="wide")
st.title("📰 Health & Wellness Blog")
set_disease_background("Image/Blog.png")
mainmenu()


# st.markdown("🛠️ This page will show health articles and updates soon.")


st.header("❤️ Heart Disease Blogs")
st.image("Image/card3.png", caption="By Dr.Aman | 2 March 2023 | Heart")
st.subheader("Stress and Heart Disease: Understanding the Connection")
st.markdown("Stress is a normal physiological response designed to protect the body from a perceived threat. However, chronic and unrelenting stress can have detrimental effects on the body, particularly the cardiovascular system, and increase the risk of heart disease.Stress hormones like adrenaline and cortisol can raise the heart rate, constrict blood vessels, and increase blood pressure, which can cause damage to the heart and blood vessels over time. ")

st.image("Image/card4.png", caption="By Dr.Rahul | 4 Dec 2022 | Diabetes")
st.subheader("Obesity and Heart Disease: What’s the Connection?")
st.markdown("Obesity and heart disease are two health conditions that are closely linked. Obesity is the accumulation of excessive body fat, which can lead to a wide range of health problems, including heart disease. It results in plaque buildup in the arteries, a condition known as atherosclerosis. Also, it increases the risk of other chronic ailments, such as hypertension and diabetes.Heart disease is a condition that affects the heart and blood vessels, often leading to heart attacks and strokes. ")




st.header("🩺 Diabetes Disease Blogs")
st.image("Image/card1.png",  caption="By Dr.Harsh | 21 Oct 2024 | Diabetes" )
st.subheader("Diabetes Reversal: Myth vs. Reality by Dr Harsh")
st.markdown("Diabetes, particularly Type 2 diabetes (T2DM), has increasingly become a focal point of research and discussion regarding its management, specifically concerning the concepts of remission and reversal. While the idea of reversing diabetes is appealing, it is crucial to understand that current scientific consensus leans towards the notion of remission rather than complete reversal. This article delves into the scientific facts surrounding diabetes remission, supported by recent studies and statistics, to clarify expectations for individuals considering this path.")



st.image("Image/card2.png", caption="By Dr.Smith | 23 June 2025 | Diabetes")
st.subheader("Understanding Your Body Clock – The Hormonal Cycle and Diabetes")
st.markdown("Understanding the diurnal hormonal cycle is essential for managing diabetes effectively, especially in a country like India, where diabetes prevalence is rising rapidly. This article explores how hormonal fluctuations throughout the day affect blood sugar levels and diabetes management, providing practical insights for individuals living with diabetes.")

st.sidebar.markdown("## 🛡️ Precautions For Diabetes patients")
st.sidebar.markdown("✅ Quit smoking now")
st.sidebar.markdown("✅ Maintain healthy weight")
st.sidebar.markdown("✅ Reduce stress levels")
st.sidebar.markdown("✅ Get enough sleep")

st.sidebar.markdown("## 🛡️ Precautions for Heart Patients")
st.sidebar.markdown("✅ Limit salt intake")
st.sidebar.markdown("✅ Monitor cholesterol levels")
st.sidebar.markdown("✅ Cut back sugar")
st.sidebar.markdown("✅ Avoid trans fats")



if st.button("🏠 Back to Home"):
    st.switch_page("app.py")
