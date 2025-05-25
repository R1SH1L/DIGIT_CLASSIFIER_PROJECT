import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

@st.cache_resource
def train_digit_classifier(X_train, y_train, **kwargs):

    default_params = {
        'n_estimators': 100,
        'random_state': 42,
        'n_jobs': -1
    }
    
    default_params.update(kwargs)
    
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    
    return accuracy, report