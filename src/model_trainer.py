"""
Model training utilities
"""

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

@st.cache_resource
def train_digit_classifier(X_train, y_train, **kwargs):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters for RandomForestClassifier
        
    Returns:
        RandomForestClassifier: Trained model
    """
    
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Update with user provided parameters
    default_params.update(kwargs)
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (accuracy, classification_report_str)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    
    return accuracy, report