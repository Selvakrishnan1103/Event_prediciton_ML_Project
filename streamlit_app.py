import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("attendance_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Event Attendance Prediction", layout="centered")

st.title("🎯 Event Attendance Prediction Using Registration Data")
st.write("Enter the attendee details to predict whether they will attend the event.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=70, value=25)

gender = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

reg_time = st.number_input("Days before registration", min_value=0, max_value=40, value=10)

distance = st.number_input("Distance from event (km)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

event_type = st.selectbox("Event Type", ["Workshop", "Seminar", "Tech Talk", "Cultural"])
event_mapping = {"Workshop": 0, "Seminar": 1, "Tech Talk": 2, "Cultural": 3}
event_type = event_mapping[event_type]

past_att = st.number_input("Past Attendance Count", min_value=0, max_value=20, value=2)

reminder = st.selectbox("Reminder Sent?", ["No", "Yes"])
reminder = 1 if reminder == "Yes" else 0

ticket = st.number_input("Ticket Price", min_value=0, max_value=2000, value=500)

weekend = st.selectbox("Is weekend event?", ["No", "Yes"])
weekend = 1 if weekend == "Yes" else 0

# Prediction button
if st.button("Predict Attendance"):
    
    input_data = np.array([[age, gender, reg_time, distance, event_type,
                            past_att, reminder, ticket, weekend]])

    # Scale numeric values
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✔ The person is **likely to attend** the event.")
    else:
        st.error("✘ The person is **unlikely to attend** the event.")
