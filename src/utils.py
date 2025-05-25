import numpy as np

def get_model_info(model):
    info = {
        'Model Type': type(model).__name__,
        'Number of Estimators': getattr(model, 'n_estimators', 'N/A'),
        'Number of Features': getattr(model, 'n_features_in_', 'N/A'),
        'Number of Classes': len(getattr(model, 'classes_', [])),
        'Classes': getattr(model, 'classes_', []).tolist() if hasattr(getattr(model, 'classes_', []), 'tolist') else list(getattr(model, 'classes_', []))
    }
    
    return info

def format_confidence(confidence):
    return f"{confidence:.1%}"

def validate_image_input(image_data):
    if image_data is None:
        return False
        
    return np.any(image_data[:, :, 3] > 0) 