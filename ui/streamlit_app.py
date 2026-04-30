import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("🧠 Sarcasm Detector")

text = st.text_area("Enter headline:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        response = requests.post(API_URL, json={"text": text})

        if response.status_code == 200:
            result = response.json()

            st.write(f"**Prediction:** {result['label']}")
            st.write(f"**Confidence:** {result['probability']:.4f}")

            if result["probability"] > 0.5:
                st.error("Sarcastic detected 😏")
            else:
                st.success("Not sarcastic 🙂")
        else:
            st.error("API error")