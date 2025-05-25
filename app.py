import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_mnist_data, get_sample_image
from model_trainer import train_digit_classifier, evaluate_model
from predictor import (
    preprocess_canvas_image, 
    preprocess_uploaded_image, 
    predict_digit,
    get_prediction_details
)
from utils import get_model_info, format_confidence, validate_image_input

st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Handwritten Digit Classifier")
st.markdown("Choose an input method and classify handwritten digits using AI!")

@st.cache_data
def get_data():
    return load_mnist_data()

@st.cache_resource
def get_model():
    X_train, _, y_train, _ = get_data()
    return train_digit_classifier(X_train, y_train)

with st.spinner("Loading data and training model..."):
    X_train, X_test, y_train, y_test = get_data()
    model = get_model()

st.success("✅ Model ready!")

st.sidebar.title("🎯 Choose Input Method")
method = st.sidebar.radio(
    "Select input type:", 
    ["Draw Digit", "MNIST Test Image", "Upload Image"]
)

if method == "Draw Digit":
    st.markdown("### ✏️ Draw a digit (0-9) in the canvas below:")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔮 Predict Digit", type="primary", use_container_width=True):
            if validate_image_input(canvas_result.image_data):
                processed_image = preprocess_canvas_image(canvas_result.image_data)
                prediction_details = get_prediction_details(model, processed_image)
                
                if prediction_details:
                    st.success(f"**Predicted Digit: {prediction_details['prediction']}**")
                    st.info(f"**Confidence: {format_confidence(prediction_details['confidence'])}**")
                    st.progress(prediction_details['confidence'])
                    
                    with st.expander("🔍 View processed image"):
                        display_img = processed_image.reshape(28, 28)
                        st.image(display_img, caption="28x28 processed", width=140)
            else:
                st.warning("⚠️ Please draw a digit first!")
    
    with col2:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.rerun()

elif method == "MNIST Test Image":
    st.markdown("### 📊 Test with MNIST dataset images:")
    
    idx = st.sidebar.slider("Select Test Image Index", 0, len(X_test) - 1, 0)
    
    image, true_label = get_sample_image(X_test, y_test, idx)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption=f"Ground Truth: {true_label}", width=200)
    
    with col2:
        prediction, confidence = predict_digit(model, X_test[idx].reshape(1, -1))
        
        st.metric("🎯 Predicted", prediction)
        st.metric("✅ Actual", true_label)
        st.metric("📊 Confidence", format_confidence(confidence))
        
        if prediction == true_label:
            st.success("✅ Correct Prediction!")
        else:
            st.error("❌ Incorrect Prediction")

elif method == "Upload Image":
    st.markdown("### 📁 Upload your own digit image:")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload PNG/JPG image", 
        type=["png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        processed_array, display_img = preprocess_uploaded_image(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(display_img, caption="Processed Image (28x28)", width=200)
        
        with col2:
            prediction, confidence = predict_digit(model, processed_array)
            
            if prediction is not None:
                st.success(f"**Predicted Digit: {prediction}**")
                st.info(f"**Confidence: {format_confidence(confidence)}**")
                st.progress(confidence)

if st.sidebar.checkbox("📈 Show Model Performance"):
    with st.spinner("Evaluating model..."):
        accuracy, report = evaluate_model(model, X_test, y_test)
        
        st.sidebar.metric("🎯 Test Accuracy", format_confidence(accuracy))
        
        with st.sidebar.expander("📋 Detailed Report"):
            st.text(report)

if st.sidebar.checkbox("ℹ️ Model Information"):
    model_info = get_model_info(model)
    
    for key, value in model_info.items():
        st.sidebar.text(f"{key}: {value}")

st.markdown("---")
st.markdown("💡 **Tips:** Draw thick, clear digits for better accuracy!")
st.markdown("Made with ❤️ using Streamlit and scikit-learn")