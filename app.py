import streamlit as st
import joblib
import torch
import os

st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")

# --- DEBUGGING: SEE WHAT THE SERVER SEES ---
st.sidebar.write("### Repository Files")
st.sidebar.write(os.listdir(".")) 

@st.cache_resource
def load_model():
    model_path = "spam_model.pkl"
    
    # 1. Check if file exists
    if not os.path.exists(model_path):
        st.error(f"❌ '{model_path}' is missing from the server.")
        return None

    # 2. Load with CPU mapping
    try:
        with open(model_path, 'rb') as f:
            return torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    except Exception:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Load failed: {e}")
            return None

model = load_model()

st.title("🛡️ Phishing Detector")

user_input = st.text_area("Paste Email Content:")

if st.button("Analyze"):
    if user_input and model:
        prediction = model.predict([user_input])[0]
        if prediction == 1:
            st.error("⚠️ Spam / Phishing")
        else:
            st.success("✅ Safe")
    elif not model:
        st.error("Model not loaded.")
    else:
        st.warning("Please enter text.")
