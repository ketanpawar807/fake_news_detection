import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("/content/model.pkl", "rb"))
vectorizer = pickle.load(open("/content/vectorizer.pkl", "rb"))

# Page setup
st.set_page_config(page_title="AI Fake News Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI-Powered Fake News Detector")
st.markdown("""
Welcome to the **Fake News Detection App**.  
Enter a news article or headline below, and this AI model will predict whether it's **Real** or **Fake**.
""")

# Input
user_input = st.text_area("📰 Paste your news content here:", height=200)

# Prediction
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        if prediction == 1:
            st.success("✅ The news seems **REAL**.")
        else:
            st.error("🚨 The news seems **FAKE**.")

st.markdown("---")
st.caption("Developed by Ketan Pawar | AI & Machine Learning Enthusiast 🤖")
