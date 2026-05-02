import os

# --- CRITICAL: FORCE CPU GLOBALLY ---
# These environment settings must be at the absolute top
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import torch
import joblib
import pandas as pd
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Phishing Detection AI",
    page_icon="🛡️",
    layout="centered"
)

# --- DIRECTORY PATHING ---
# This finds the exact folder where app.py is stored on the Streamlit server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "spam_model.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

# --- ADVANCED MODEL LOADING ---
@st.cache_resource
def load_model_safely():
    """
    Handles LFS files and forces CUDA tensors to load on CPU hardware.
    """
    # 1. Check if file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ '{MODEL_NAME}' not found in the root directory.")
        # Diagnostic: Show the server's files to help find it
        st.info(f"Available files: {os.listdir(BASE_DIR)}")
        return None

    # 2. Check if the file is an LFS pointer (too small)
    if os.path.getsize(MODEL_PATH) < 2000:
        st.error("❌ The model file appears to be a Git LFS pointer (only 1KB).")
        st.warning("Please ensure Git LFS is properly configured and synced to GitHub.")
        return None

    # 3. Attempt to load with CPU mapping
    try:
        # Most modern torch-based pickles need weights_only=False for complex pipelines
        return torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        try:
            # Fallback for standard Scikit-learn/joblib files
            return joblib.load(MODEL_PATH)
        except Exception as final_e:
            st.error("❌ Critical Loading Error")
            st.code(f"Technical Log: {str(final_e)}")
            return None

# Initialize Model
model = load_model_safely()

# --- USER INTERFACE ---
st.title("🛡️ Phishing Email Detector")
st.markdown("""
    This app analyzes email text to detect phishing attempts or spam. 
    It is currently running in **CPU-Compatible Mode**.
""")

# Input Area
user_input = st.text_area("Paste the email or message content here:", height=250, placeholder="Example: Dear customer, your account is suspended. Click here to verify...")

# Analysis Logic
if st.button("Run Analysis", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    elif model is None:
        st.error("The model could not be loaded. Please check the sidebar/error messages.")
    else:
        with st.spinner("Analyzing message patterns..."):
            try:
                # Prediction
                # Most NLP models expect a list/array of strings
                prediction = model.predict([user_input])[0]
                
                st.divider()
                
                # Assume 1 = Phishing/Spam, 0 = Safe
                if prediction == 1:
                    st.error("### ⚠️ Result: Potential Phishing Attempt")
                    st.write("Our AI detected patterns consistent with fraudulent or spam messages.")
                else:
                    st.success("### ✅ Result: Message Appears Safe")
                    st.write("No significant phishing indicators were found in this text.")
                
                # Show Confidence if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([user_input])[0]
                    confidence = max(probs)
                    st.progress(float(confidence))
                    st.caption(f"Analysis Confidence: {confidence:.2%}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# --- FOOTER ---
st.divider()
st.caption("Developed for Academic Research • Powered by Streamlit & PyTorch")
