# Handwritten Digit Prediction

A beginner-friendly machine learning project that predicts handwritten digits from uploaded images.

## Features

- Streamlit web app for image upload and prediction
- Random Forest classifier trained on the scikit-learn digits dataset
- Top 3 prediction probabilities
- Validation accuracy and confusion matrix
- Original notebook preserved under `notebooks/`

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- notebooks/
|   `-- handwritten_digit_prediction.ipynb
`-- src/
    |-- __init__.py
    `-- digit_model.py
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. The app trains a Random Forest model on the scikit-learn digits dataset if no saved model exists.
2. Uploaded images are converted to grayscale.
3. The image is resized to `8x8`, normalized, and classified.
4. The app displays the predicted digit and confidence scores.

## Future Improvements

- Add a drawing canvas for live digit input.
- Train a CNN on MNIST for stronger performance.
- Add sample test images.
- Deploy the app on Streamlit Community Cloud.
