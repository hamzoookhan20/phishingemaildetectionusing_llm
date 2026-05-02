import streamlit as st
import joblib
import torch
import os
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️")

# --- ADVANCED CPU LOADING STRATEGY ---
@st.cache_resource
def load_model_safely():
    model_path = "spam_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ '{model_path}' not found. Please check GitHub LFS.")
        return None

    try:
        # Strategy A: Use torch.load with explicit CPU mapping
        # This is the most reliable way to bypass the CUDA error
        with open(model_path, 'rb') as f:
            return torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        try:
            # Strategy B: If Strategy A fails, try joblib with a context manager
            # Some sklearn pipelines need this
            return joblib.load(model_path)
        except Exception as final_e:
            st.error(f"❌ All loading attempts failed.")
            st.code(f"Technical Details: {str(final_e)}")
            return None

# Load the model
model = load_model_safely()

# --- UI ---
st.title("🛡️ Phishing Email Detection")
st.write("Enter text below to analyze for spam or phishing indicators.")

user_input = st.text_area("Email Content", height=200)

if st.button("Analyze", type="primary"):
    if user_input.strip():
        if model:
            try:
                # Note: Most models expect a list of strings
                prediction = model.predict([user_input])[0]
                
                st.divider()
                if prediction == 1:
                    st.error("### ⚠️ Result: Phishing/Spam Detected")
                else:
                    st.success("### ✅ Result: Safe Message")
                
                # Show probability if available
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba([user_input]).max()
                    st.info(f"**Confidence:** {prob:.2%}")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model is not loaded.")
    else:
        st.warning("Please enter a message.")

# Footer Debugger (Optional)
if not os.path.exists("spam_model.pkl"):
    st.sidebar.warning("Model file missing from root directory.")
else:
    st.sidebar.success("Model file detected in root directory.")
