import streamlit as st
import joblib

model = joblib.load("knn_model.pkl")

st.title("BMI Class Prediction")


weight = st.number_input("Enter Weight(x2) (kg)")
height = st.number_input("Enter Height(y2) (cm)")

if st.button("Predict"):
    result = model.predict([[weight, height]])
    
    if result == 0:
        st.success("Underweight")
    elif result == 1:
        st.success("Normal")
    else:
        st.success("Overweight")