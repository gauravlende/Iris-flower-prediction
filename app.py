import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("model.pkl")

st.title("Iris Flower Prediction App")

# User Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)