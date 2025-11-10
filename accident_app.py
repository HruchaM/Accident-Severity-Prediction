import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("accident_model.pkl")

st.title("🚗 Accident Severity Prediction App")

st.write("Enter details below to predict whether a person is likely to be **Injured** or **Killed** in an accident:")

# Example input fields
person_age = st.number_input("Person's Age", min_value=1, max_value=100, value=30)
person_type = st.selectbox("Person Type", ["Occupant", "Pedestrian"])
emotional_status = st.selectbox("Emotional Status", ["Conscious", "Unconscious", "Apparent Death"])
safety_equipment = st.selectbox("Safety Equipment", ["Lap Belt & Harness", "Unknown"])
complaint = st.selectbox("Complaint", ["Internal", "Severe Bleeding", "Complaint of Pain", "Complaint of Pain or Nausea"])
bodily_injury = st.selectbox("Bodily Injury", ["Head", "Unknown"])
ejection = st.selectbox("Ejection", ["Not Ejected", "Unknown"])

# Create dataframe for input
input_data = pd.DataFrame({
    'PERSON_AGE': [person_age],
    'PERSON_TYPE': [person_type],
    'EMOTIONAL_STATUS': [emotional_status],
    'SAFETY_EQUIPMENT': [safety_equipment],
    'COMPLAINT': [complaint],
    'BODILY_INJURY': [bodily_injury],
    'EJECTION': [ejection]
})

# Predict
if st.button("Predict Severity"):
    prediction = model.predict(input_data)[0]
    result = "🟢 Injured" if prediction == 0 else "🔴 Killed"
    st.subheader(f"Predicted Outcome: {result}")
