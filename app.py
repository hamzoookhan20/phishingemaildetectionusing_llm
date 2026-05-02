import streamlit as st
import joblib
import torch
import os
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Phishing Detection AI",
    page_icon="🛡️",
    layout="centered"
)

# --- FIXED MODEL LOADING LOGIC ---
@st.cache_resource
def load_model():
    """
    Forces the model to load on CPU regardless of how it was saved.
    """
    model_filename = "spam_model.pkl"
    
    if not os.path.exists(model_filename):
        st.error(f"❌ File '{model_filename}' not found in repository.")
        return None

    try:
        # Since the error is explicitly about CUDA, we use torch.load directly.
        # torch.load can open most .pkl files that contain torch tensors.
        with open(model_filename, 'rb') as f:
            # The 'map_location' parameter is the fix for your specific error.
            return torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        try:
            # Secondary fallback: If it's a pure Scikit-Learn model wrapped in joblib
            return joblib.load(model_filename)
        except Exception as e2:
            st.error(f"❌ Final attempt failed. Original Error: {e}")
            return None

# Initialize the model
model = load_model()

# --- USER INTERFACE ---
st.title("🛡️ Phishing Email Detector")
st.markdown("""
Paste the content of a suspicious email or message below. 
The AI will force-load the model onto the CPU for analysis.
""")

# Input Area
message_text = st.text_area("Message Content", placeholder="Paste email text here...", height=200)

# Prediction Action
if st.button("Run Analysis", type="primary"):
    if not message_text.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None:
        st.error("Model load failed. Check the error message above.")
    else:
        with st.spinner("Analyzing text patterns on CPU..."):
            try:
                # Prediction logic
                # If your model is a pipeline/Transformer, it expects a list
                prediction = model.predict([message_text])[0]
                
                st.divider()
                
                if prediction == 1:
                    st.error("### ⚠️ High Risk Detected")
                    st.write("This message matches known phishing signatures.")
                else:
                    st.success("### ✅ Appears Safe")
                    st.write("No significant phishing indicators found.")
                
                # Confidence display
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([message_text])[0]
                    confidence = max(probs)
                    st.progress(float(confidence))
                    st.caption(f"Model Confidence: {confidence:.2%}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# --- FOOTER ---
st.divider()
st.caption("Built with Streamlit • CPU-compatible Mode")
