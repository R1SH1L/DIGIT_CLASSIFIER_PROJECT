import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import streamlit as st

@st.cache_data
def load_mnist_data(test_size=0.2, random_state=42):
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_sample_image(X_test, y_test, index):
    image = X_test[index].reshape(28, 28)
    true_label = y_test[index]
    
    return image, true_label