# Handwritten Digit Prediction

A beginner-friendly machine learning web app that classifies handwritten digits from uploaded images using a Random Forest classifier.

## Features

- Streamlit web app with dark theme
- Upload any digit image (PNG / JPG)
- Random Forest classifier trained on scikit-learn's digits dataset
- Top 3 prediction probabilities with confidence scores
- Validation accuracy and confusion matrix
- Original Jupyter notebook preserved under `notebooks/`

## Tech Stack

- Python 3
- Streamlit
- scikit-learn
- Pillow / NumPy
- Jupyter Notebook (development)

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── handwritten_digit_prediction.ipynb
└── src/
    ├── __init__.py
    └── digit_model.py
```

## Setup — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Sriram127/hand-written_digits_prediction.git
cd hand-written_digits_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## How It Works

1. The app trains a Random Forest model on scikit-learn's 8×8 pixel digits dataset on startup.
2. You upload a digit image (any size — it will be resized automatically).
3. The image is converted to grayscale, resized to 8×8, and normalised.
4. The model returns the predicted digit and the top 3 confidence scores.

## Future Improvements

- Add an interactive drawing canvas so users can draw digits directly in the browser
- Train a CNN on MNIST for significantly higher accuracy
- Add sample test images to the repo for quick verification
