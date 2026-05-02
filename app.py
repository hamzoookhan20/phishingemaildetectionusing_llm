import os
import torch
import streamlit as st
import joblib

# --- THE CUDA-TO-CPU MONKEY PATCH ---
# This MUST come before any other torch logic
class CPU_Unpickler(joblib.numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

# Global override
def force_cpu_load(path):
    return torch.load(path, map_location=torch.device('cpu'), weights_only=False)

# --- PAGE SETUP ---
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")

@st.cache_resource
def load_model():
    model_path = "spam_model.pkl"
    
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None

    try:
        # Strategy 1: Direct Torch Load with Map Location
        return torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        try:
            # Strategy 2: If it's a specialized joblib/torch hybrid
            # We use the map_location string directly
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Critical Error: {e}")
            return None

model = load_model()

# --- APP INTERFACE ---
st.title("🛡️ Phishing Email Detector")
st.info("System Status: CPU-Mode Active")

user_input = st.text_area("Paste Email Text:", height=200)

if st.button("Analyze", type="primary"):
    if user_input and model:
        try:
            # Ensure input is a list
            prediction = model.predict([user_input])[0]
            st.divider()
            if prediction == 1:
                st.error("### ⚠️ Result: Phishing/Spam")
            else:
                st.success("### ✅ Result: Safe")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
