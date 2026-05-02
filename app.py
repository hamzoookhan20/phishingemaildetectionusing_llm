import os
import io
import torch
import streamlit as st
import joblib
import pickle

# --- 1. GLOBAL ENVIRONMENT OVERRIDE ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- 2. THE ADVANCED CPU UNPICKLER ---
# This class intercepts the "Pickle" process and forces CUDA storages to CPU
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

@st.cache_resource
def load_model_final():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "spam_model.pkl")
    
    if not os.path.exists(model_path):
        return None

    try:
        # Strategy A: Use the Custom Unpickler (The most aggressive fix)
        with open(model_path, 'rb') as f:
            return CPU_Unpickler(f).load()
    except Exception:
        try:
            # Strategy B: Standard torch load with weights_only=False
            return torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            try:
                # Strategy C: Joblib fallback
                return joblib.load(model_path)
            except Exception as final_e:
                st.error(f"Internal Loading Error: {final_e}")
                return None

# --- 3. PAGE SETUP & UI ---
st.set_page_config(page_title="Phishing Detection AI", page_icon="🛡️")
st.title("🛡️ Phishing Email Detector")

model = load_model_final()

if model is not None:
    st.success("✅ Model Successfully Mapped to CPU")
    
    user_input = st.text_area("Paste message content:", height=200)
    
    if st.button("Analyze Message", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Most NLP models expect a list of strings: [text]
                    prediction = model.predict([user_input])[0]
                    
                    st.divider()
                    if prediction == 1 or str(prediction).lower() == 'spam':
                        st.error("### ⚠️ Result: Potential Phishing/Spam")
                    else:
                        st.success("### ✅ Result: Message Appears Safe")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.warning("Please enter some text.")
else:
    st.error("❌ Critical: 'spam_model.pkl' could not be initialized.")

# --- 4. SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.write("### System Info")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    
    model_path = os.path.join(os.path.dirname(__file__), "spam_model.pkl")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.write(f"Model File Found: Yes")
        st.write(f"Model Size: {size_mb:.2f} MB")
