
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction App")

# ===============================
# Load Model & Tools
# ===============================
try:
    model = joblib.load("model/titanic_model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    gender_encoder = joblib.load("model/gender_encoder.joblib")
    embarked_encoder = joblib.load("model/embarked_encoder.joblib")
except FileNotFoundError:
    st.error("Error: Model or preprocessor files not found. Please ensure the training notebook has been run.")
    st.stop()

# ===============================
# User Inputs
# ===============================
pclass_options = {1: '1st Class', 2: '2nd Class', 3: '3rd Class'}
pclass_selected = st.selectbox("Passenger Class", options=list(pclass_options.keys()), format_func=lambda x: pclass_options[x])

gender_options = ["male", "female"]
gender_selected = st.selectbox("Gender", gender_options)

age = st.slider("Age", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
family = st.slider("Number of Family Members (Siblings/Spouses/Parents/Children)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid ($)", min_value=0.0, value=30.0)

embarked_options = ["S", "C", "Q"]
embarked_selected = st.selectbox("Port of Embarkation", embarked_options)

# Encode inputs
gender_encoded = gender_encoder.transform([gender_selected])[0]
embarked_encoded = embarked_encoder.transform([embarked_selected])[0]

# Prepare input for scaling and prediction
input_data = np.array([[pclass_selected, gender_encoded, age, family, fare, embarked_encoded]])
input_scaled = scaler.transform(input_data)

# ===============================
# Prediction
# ===============================
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] # Probability of surviving

    if prediction == 1:
        st.success(f"🎉 Prediction: Passenger SURVIVED! (Probability: {probability:.2f})")
        st.balloons()
    else:
        st.error(f"💀 Prediction: Passenger DID NOT SURVIVE. (Probability: {probability:.2f})")
