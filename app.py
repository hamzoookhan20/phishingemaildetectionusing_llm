import os
import torch
import streamlit as st
import joblib

# --- 1. FORCE CPU AT THE START ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")

# --- 2. DEBUGGING SIDEBAR ---
st.sidebar.title("System Diagnostics")
all_files = os.listdir(".")
st.sidebar.write("Files detected:", all_files)

# --- 3. MODEL LOADING LOGIC ---
@st.cache_resource
def load_phishing_model():
    model_path = "spam_model.pkl"
    
    if not os.path.exists(model_path):
        return None

    # Handle the CUDA-to-CPU error automatically
    try:
        return torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"Load Error: {e}")
            return None

model = load_phishing_model()

# --- 4. USER INTERFACE ---
st.title("🛡️ Phishing Email Detector")

if model is None:
    st.error("❌ Model file ('spam_model.pkl') not found in the repository.")
    st.info("Check if Git LFS successfully uploaded the 268MB file to GitHub.")
else:
    st.success("✅ Model loaded successfully on CPU.")
    
    user_input = st.text_area("Paste the email content to analyze:", height=200)
    
    if st.button("Analyze Message", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Most models expect a list of strings
                    prediction = model.predict([user_input])[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error("### ⚠️ Result: Phishing/Spam")
                    else:
                        st.success("### ✅ Result: Safe")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter some text.")
