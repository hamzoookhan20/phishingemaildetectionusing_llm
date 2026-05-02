import os
import io
import torch
import streamlit as st
import joblib

# --- STEP 1: GLOBAL ENVIRONMENT OVERRIDE ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- STEP 2: THE "HEAVY DUTY" CPU UNPICKLER ---
# This class redefines how 'torch' objects are rebuilt from the file
class CPU_Unpickler(io.BytesIO):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

@st.cache_resource
def load_model_final():
    model_path = "spam_model.pkl"
    
    try:
        # Strategy A: Try the most direct torch load
        return torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception:
        try:
            # Strategy B: Manual byte-stream redirection
            with open(model_path, 'rb') as f:
                return torch.load(f, map_location='cpu')
        except Exception as e:
            # Strategy C: Joblib load (for Sklearn wrappers)
            try:
                return joblib.load(model_path)
            except Exception as final_e:
                st.error(f"Critical loading failure: {final_e}")
                return None

# --- STEP 3: USER INTERFACE ---
st.set_page_config(page_title="Phishing Detection AI", page_icon="🛡️")

st.title("🛡️ Phishing Email Detector")
st.caption("Status: Model detected. Forcing CPU execution...")

model = load_model_final()

if model is not None:
    st.success("✅ Model successfully mapped to CPU.")
    
    user_input = st.text_area("Paste message content:", height=200, placeholder="Enter text here...")
    
    if st.button("Analyze Message", type="primary"):
        if user_input.strip():
            with st.spinner("Processing..."):
                try:
                    # NLP models usually require input as a list: [text]
                    prediction = model.predict([user_input])[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error("### ⚠️ Result: Phishing/Spam Detected")
                        st.warning("This message contains high-risk patterns.")
                    else:
                        st.success("### ✅ Result: Message Appears Safe")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.warning("Please enter some text first.")
else:
    st.error("Model mapping failed. Please check the logs.")

# --- SIDEBAR DIAGNOSTICS ---
with st.sidebar:
    st.write("### System Info")
    st.write(f"CUDA Available: {torch.cuda.is_available()}")
    st.write(f"Model Size: {os.path.getsize('spam_model.pkl') / (1024*1024):.2f} MB")
