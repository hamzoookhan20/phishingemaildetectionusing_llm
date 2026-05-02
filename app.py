import os
import io
import torch
import streamlit as st
import joblib

# --- 1. GLOBAL ENVIRONMENT OVERRIDE ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- 2. THE CPU UNPICKLER ---
class CPU_Unpickler(io.BytesIO):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

@st.cache_resource
def load_model_final():
    # Use absolute path to avoid "File Not Found" errors
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "spam_model.pkl")
    
    if not os.path.exists(model_path):
        return None

    try:
        # Priority 1: Direct Torch Load with CPU mapping
        return torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception:
        try:
            # Priority 2: Joblib fallback
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Internal Loading Error: {e}")
            return None

# --- 3. PAGE SETUP & UI ---
st.set_page_config(page_title="Phishing Detection AI", page_icon="🛡️")
st.title("🛡️ Phishing Email Detector")

model = load_model_final()

if model is not None:
    st.success("✅ Model Loaded Successfully")
    
    user_input = st.text_area("Paste message content:", height=200)
    
    if st.button("Analyze Message", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Prediction (Wrapped in list for NLP pipelines)
                    prediction = model.predict([user_input])[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error("### ⚠️ Result: Potential Phishing/Spam")
                    else:
                        st.success("### ✅ Result: Message Appears Safe")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.warning("Please enter some text.")
else:
    st.error("❌ Critical: 'spam_model.pkl' could not be initialized.")
    st.info("Ensure the file is in the root directory and pushed via Git LFS.")

# --- 4. SAFE SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.write("### System Info")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Safe file size check
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "spam_model.pkl")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.write(f"Model File Found: Yes")
        st.write(f"Model Size: {size_mb:.2f} MB")
    else:
        st.write("Model File Found: No")
