# news_predictor.py
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\models")

# Load model once
model = joblib.load(MODEL_PATH)

def predict_news(text: str) -> str:
    """Return 'Fake' or 'Real' prediction for the given text"""
    pred = model.predict([text])[0]
    return "Fake" if pred == 1 else "Real"