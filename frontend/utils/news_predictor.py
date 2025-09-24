import os
import joblib

def load_model(model_path=r"C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\models\\fake_news_model.pkl"):
    """
    Load a trained ML model from the models folder.
    Default: fake_news_model.pkl
    """

    # Path of this file (utils/)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Go up TWO levels (utils -> frontend -> project root)
    project_root = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

    # Path to models folder
    models_dir = os.path.join(project_root, "models")

    # Full path to model
    model_path = os.path.join(models_dir, "C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\models\\fake_news_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    print(f"✅ Loading model from: {model_path}")
    return joblib.load(model_path)