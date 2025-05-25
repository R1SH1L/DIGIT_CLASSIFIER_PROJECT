"""
Utility functions for the digit classifier
"""

import numpy as np

def get_model_info(model):
    """
    Get information about the trained model
    
    Args:
        model: Trained scikit-learn model
        
    Returns:
        dict: Model information
    """
    info = {
        'Model Type': type(model).__name__,
        'Number of Estimators': getattr(model, 'n_estimators', 'N/A'),
        'Number of Features': getattr(model, 'n_features_in_', 'N/A'),
        'Number of Classes': len(getattr(model, 'classes_', [])),
        'Classes': getattr(model, 'classes_', []).tolist() if hasattr(getattr(model, 'classes_', []), 'tolist') else list(getattr(model, 'classes_', []))
    }
    
    return info

def format_confidence(confidence):
    """
    Format confidence score for display
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Formatted confidence string
    """
    return f"{confidence:.1%}"

def validate_image_input(image_data):
    """
    Validate if image data contains drawing
    
    Args:
        image_data: Canvas image data
        
    Returns:
        bool: True if image contains drawing
    """
    if image_data is None:
        return False
        
    # Check if there's any drawing (non-white pixels)
    return np.any(image_data[:, :, 3] > 0)  # Check alpha channel