import streamlit as st
import joblib
import torch
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Phishing & Spam Detection",
    page_icon="📧",
    layout="centered"
)

# 2. Optimized Model Loading Function
@st.cache_resource
def load_phishing_model():
    model_path = "spam_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Critical Error: '{model_path}' not found in the repository.")
        return None

    try:
        # Step 1: Attempt standard load
        # This works if the pickle was saved with joblib and is CPU-compatible
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # Step 2: Fallback for GPU-trained models (The CPU Mapping Fix)
        # If the pickle contains PyTorch tensors trained on CUDA, we force them to CPU
        try:
            with open(model_path, 'rb') as f:
                # Weights_only=False is used because .pkl files often contain custom classes
                return torch.load(f, map_location=torch.device('cpu'), weights_only=False)
        except Exception as final_error:
            st.error(f"Failed to load model: {str(final_error)}")
            return None

# Initialize Model
model = load_phishing_model()

# 3. Streamlit User Interface
st.title("📧 Phishing Email Detection")
st.markdown("""
    This application uses a trained model to analyze messages for potential security threats. 
    **Paste the content of an email or SMS below to check its safety.**
""")

# Input Area
user_input = st.text_area("Message Content", placeholder="Enter the email text here...", height=250)

# 4. Prediction Logic
if st.button("Analyze Message", type="primary"):
    if user_input.strip():
        if model is not None:
            with st.spinner("Processing large model file..."):
                try:
                    # Most Scikit-learn/NLP pipelines expect a list of strings
                    # We wrap user_input in [ ] to match the expected input shape
                    prediction = model.predict([user_input])[0]
                    
                    # Optional: Handling probability if your model supports predict_proba
                    confidence = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba([user_input])
                        confidence = proba.max()

                    st.divider()
                    
                    # Displaying results based on common label logic (1=Spam, 0=Safe)
                    if prediction == 1:
                        st.error("### ⚠️ Potential Phishing or Spam Detected")
                        if confidence:
                            st.metric("Confidence Level", f"{confidence:.2%}")
                        st.warning("Exercise caution: This message contains patterns common in phishing attacks.")
                    else:
                        st.success("### ✅ This Message Appears to be Safe")
                        if confidence:
                            st.metric("Confidence Level", f"{confidence:.2%}")
                        st.info("The model did not find high-risk phishing indicators in this text.")
                
                except Exception as eval_error:
                    st.error(f"Prediction Error: {str(eval_error)}")
        else:
            st.error("Model is not available. Please check the logs.")
    else:
        st.warning("Please provide a message to analyze.")

# Footer
st.caption("Note: This is an AI-powered tool. Always verify suspicious links manually.")
