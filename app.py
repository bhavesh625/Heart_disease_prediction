import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Heart Disease Prediction App")
st.write("Fill the patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [0, 1])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs,
                                restecg, thalach, exang]],
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                         'restecg', 'thalach', 'exang'])
    result = model.predict(input_data)[0]
    st.success("Prediction: " + ("⚠️ Heart Disease Detected" if result == 1 else "✅ No Heart Disease"))
