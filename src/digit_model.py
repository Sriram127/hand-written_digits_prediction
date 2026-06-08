from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "digit_classifier.joblib"


def train_model():
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    return model, metrics


def load_or_train_model():
    if MODEL_PATH.exists():
        payload = joblib.load(MODEL_PATH)
        return payload["model"], payload["metrics"]
    return train_model()


def prepare_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)

    if pixels.max() > 0:
        pixels = pixels / pixels.max() * 16.0

    return pixels.reshape(1, -1)


def predict_digit(image):
    model, _metrics = load_or_train_model()
    features = prepare_image(image)
    probabilities = model.predict_proba(features)[0]
    prediction = int(np.argmax(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        {"digit": int(index), "confidence": float(probabilities[index])}
        for index in top_indices
    ]
    return prediction, top_predictions
