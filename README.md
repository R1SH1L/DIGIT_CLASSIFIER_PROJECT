# Handwritten Digit Classifier

A web application that classifies handwritten digits (0-9) using machine learning. Built with Streamlit and scikit-learn.

## Project Structure

```
digit-classifier/
├── src/
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading utilities
│   ├── model_trainer.py   # Model training logic
│   ├── predictor.py       # Prediction utilities
│   └── utils.py          # Helper functions
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md            # Documentation
└── .gitignore          # Git ignore rules
```

## Setup Instructions

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the application:
    ```bash
    streamlit run app.py
    ```

## Features

- **Draw Digit**: Interactive canvas for drawing digits
- **MNIST Test**: Test with real MNIST dataset images
- **Upload Image**: Upload your own digit images
- **Model Performance**: View accuracy and detailed metrics
- **Model Information**: See model details and parameters

## Model Details

- **Algorithm**: Random Forest Classifier
- **Dataset**: MNIST (70,000 handwritten digits)
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- **Accuracy**: ~97%
- **Prediction Time**: <100ms
- **Input**: 28x28 grayscale images

## Performance Metrics

| Metric | Value |
|--------|-------|
| Precision | 97% |
| Recall | 97% |
| F1-Score | 97% |
| Support | 10,000 |


## System Requirements

- Python 3.8+
- 2GB RAM minimum
- Modern web browser

## Technologies Used

- **Streamlit**: Web framework
- **scikit-learn**: Machine learning
- **PIL**: Image processing
- **NumPy**: Numerical computing