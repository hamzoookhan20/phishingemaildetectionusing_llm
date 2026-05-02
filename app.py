import streamlit as st
import joblib
import torch
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Phishing Detection AI",
    page_icon="🛡️",
    layout="centered"
)

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_model():
    """
    Loads the 268MB model file. 
    Handles the GPU-to-CPU mapping automatically.
    """
    model_filename = "spam_model.pkl"
    
    # Verify file existence
    if not os.path.exists(model_filename):
        # This helpfully lists files for you in the UI if it fails
        available_files = os.listdir(".")
        st.error(f"❌ File '{model_filename}' not found.")
        st.info(f"Files currently in directory: {available_files}")
        return None

    try:
        # Attempt standard load
        return joblib.load(model_filename)
    except Exception:
        try:
            # Fallback: Load GPU-trained model onto CPU
            with open(model_filename, 'rb') as f:
                return torch.load(f, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

# Initialize the model
model = load_model()

# --- USER INTERFACE ---
st.title("🛡️ Phishing Email Detector")
st.markdown("""
Paste the content of a suspicious email or message below. 
Our AI will analyze the text for patterns common in phishing and spam.
""")

# Input Area
message_text = st.text_area("Message Content", placeholder="Paste email text here...", height=200)

# Prediction Action
if st.button("Run Analysis", type="primary"):
    if not message_text.strip():
        st.warning("Please enter some text to analyze.")
    elif model is None:
        st.error("The model is not loaded. Check your file name and GitHub LFS status.")
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                # Wrap input in a list as most sklearn/NLP models expect
                prediction = model.predict([message_text])[0]
                
                st.divider()
                
                # Result Logic (Assumes 1 = Spam/Phishing, 0 = Safe)
                if prediction == 1:
                    st.error("### ⚠️ High Risk Detected")
                    st.write("This message matches known phishing or spam signatures.")
                else:
                    st.success("### ✅ Appears Safe")
                    st.write("The AI did not find significant phishing indicators in this message.")
                
                # Display probability if the model supports it
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([message_text])[0]
                    confidence = max(probs)
                    st.progress(float(confidence))
                    st.caption(f"Model Confidence: {confidence:.2%}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# --- FOOTER ---
st.divider()
st.caption("Built with Streamlit • Model: 268MB DistilBERT/Sklearn Hybrid")
