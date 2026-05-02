import os

# --- THE FIX: FORCE CPU GLOBALLY ---
# These must stay at the VERY top of your file, before any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
# Manually overriding the default loading behavior
torch.set_default_device('cpu')

import streamlit as st
import joblib
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")

@st.cache_resource
def load_model_safely():
    model_path = "spam_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found. Ensure LFS is synced.")
        return None

    try:
        # Strategy: Use torch.load with map_location AND weights_only=False
        # This handles the complex objects often found in .pkl files
        model = torch.load(
            model_path, 
            map_location=torch.device('cpu'), 
            weights_only=False
        )
        return model
    except Exception as e:
        st.sidebar.info("Standard torch.load failed, trying joblib fallback...")
        try:
            # Fallback for pure Sklearn wrappers
            return joblib.load(model_path)
        except Exception as final_e:
            st.error("❌ Critical: All loading attempts failed.")
            st.code(f"Final Error: {str(final_e)}")
            return None

# Load the model
model = load_model_safely()

# --- APP UI ---
st.title("🛡️ Phishing Email Detection")
st.write("Force-loading model to CPU environment...")

user_input = st.text_area("Paste Email Content here:", height=200)

if st.button("Analyze Now", type="primary"):
    if user_input.strip() and model:
        with st.spinner("Analyzing..."):
            try:
                # Prediction logic
                prediction = model.predict([user_input])[0]
                
                st.divider()
                if prediction == 1:
                    st.error("### ⚠️ Result: Phishing/Spam Detected")
                else:
                    st.success("### ✅ Result: Safe Message")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    elif not model:
        st.error("Model is not loaded. Check technical details above.")
    else:
        st.warning("Please enter some text.")
