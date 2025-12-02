import os
import joblib

THIS_DIR = os.path.dirname(__file__)
FIELD_MODEL_PATH = os.path.join(THIS_DIR, "field_model.pkl")
VECTORIZER_PATH = os.path.join(THIS_DIR, "resume_vectorizer.pkl")

_model = None
_vectorizer = None

def _load():
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(FIELD_MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)

def predict_field(resume_text: str):
    """Predict career field using trained Resume.csv model."""
    _load()
    X = _vectorizer.transform([resume_text])
    probs = _model.predict_proba(X)[0]

    idx = probs.argmax()
    label = _model.classes_[idx]
    confidence = float(probs[idx])

    return label, confidence
