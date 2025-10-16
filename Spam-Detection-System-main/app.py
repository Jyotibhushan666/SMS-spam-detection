import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model files: {e}")

st.title("📩 SMS Spam Detection System")

# User input
user_input = st.text_area("Enter the message:")

if st.button("Check Spam"):
    if user_input:
        # Preprocess and predict
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)

        # Output
        if prediction[0] == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("Please enter a message first.")
