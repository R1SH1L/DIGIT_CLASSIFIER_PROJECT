"""
Prediction and image preprocessing utilities
"""

import numpy as np
from PIL import Image, ImageOps

def preprocess_canvas_image(canvas_data):
    """
    Preprocess image from drawing canvas
    
    Args:
        canvas_data: Raw canvas image data
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    if canvas_data is None:
        return None
        
    # Convert to grayscale and invert colors
    img = Image.fromarray((255 - canvas_data[:, :, 0]).astype(np.uint8))
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).reshape(1, -1) / 255.0
    
    return img_array

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess uploaded image file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (processed_image_array, pil_image_for_display)
    """
    # Open and convert to grayscale
    img = Image.open(uploaded_file).convert("L")
    
    # Invert colors (white digit on black background)
    img = ImageOps.invert(img)
    
    # Resize to 28x28
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(img_resized).reshape(1, -1) / 255.0
    
    return img_array, img_resized

def predict_digit(model, image_array):
    """
    Predict digit from preprocessed image
    
    Args:
        model: Trained classifier
        image_array: Preprocessed image array
        
    Returns:
        tuple: (predicted_digit, confidence_score)
    """
    if image_array is None:
        return None, None
        
    prediction = model.predict(image_array)[0]
    probabilities = model.predict_proba(image_array)[0]
    confidence = np.max(probabilities)
    
    return int(prediction), float(confidence)

def get_prediction_details(model, image_array):
    """
    Get detailed prediction information
    
    Args:
        model: Trained classifier
        image_array: Preprocessed image array
        
    Returns:
        dict: Prediction details including top predictions
    """
    if image_array is None:
        return None
        
    probabilities = model.predict_proba(image_array)[0]
    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [(int(idx), float(probabilities[idx])) for idx in top_indices]
    
    return {
        'prediction': int(prediction),
        'confidence': float(confidence),
        'top_predictions': top_predictions,
        'all_probabilities': probabilities.tolist()
    }