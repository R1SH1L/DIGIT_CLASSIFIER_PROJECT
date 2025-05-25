import numpy as np
from PIL import Image, ImageOps

def preprocess_canvas_image(canvas_data):
    if canvas_data is None:
        return None
        
    img = Image.fromarray((255 - canvas_data[:, :, 0]).astype(np.uint8))
    
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img).reshape(1, -1) / 255.0
    
    return img_array

def preprocess_uploaded_image(uploaded_file):

    img = Image.open(uploaded_file).convert("L")
    
    img = ImageOps.invert(img)
    
    img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    img_array = np.array(img_resized).reshape(1, -1) / 255.0
    
    return img_array, img_resized

def predict_digit(model, image_array):
    if image_array is None:
        return None, None
        
    prediction = model.predict(image_array)[0]
    probabilities = model.predict_proba(image_array)[0]
    confidence = np.max(probabilities)
    
    return int(prediction), float(confidence)

def get_prediction_details(model, image_array):
    if image_array is None:
        return None
        
    probabilities = model.predict_proba(image_array)[0]
    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = [(int(idx), float(probabilities[idx])) for idx in top_indices]
    
    return {
        'prediction': int(prediction),
        'confidence': float(confidence),
        'top_predictions': top_predictions,
        'all_probabilities': probabilities.tolist()
    }