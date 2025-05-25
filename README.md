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
- **Accuracy**: ~97%
- **Input**: 28x28 grayscale images

## Technologies Used

- **Streamlit**: Web framework
- **scikit-learn**: Machine learning
- **PIL**: Image processing
- **NumPy**: Numerical computing

## Deployment

This application is ready for deployment on Streamlit Cloud. Simply connect your GitHub repository to Streamlit Cloud and deploy.