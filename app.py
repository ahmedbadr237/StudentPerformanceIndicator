import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import os
import sys
from src.exception import CustomException

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #ff2020;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>📚 Student Performance Prediction 🎓</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Enter the student details to predict the Math Score</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("🧑 :red[Gender]", ["male", "female"])
    lunch = st.selectbox("🍽 :red[Lunch Type]", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("📖 :red[Test Preparation Course]", ["none", "completed"])

with col2:
    race_ethnicity = st.selectbox("🌎 :red[Race/Ethnicity]", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("🎓 :red[Parental Education]", 
                                               ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    reading_score = st.number_input("📚 :red[Reading Score]", min_value=0, max_value=100, value=50)

if st.button("🔮 Predict Math Score"):
    try:
        CustomDataObj = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score
        )
        
        data = CustomDataObj.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        predicted_score = predict_pipeline.predict(data)
        
        st.success(f"🎯 :red[Predicted Math Score: **{round(float(predicted_score),2)}** 🎯]")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
