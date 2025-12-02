import os
import re
import joblib
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "Resume.csv")

FIELD_MODEL_PATH = os.path.join(THIS_DIR, "field_model.pkl")
VECTORIZER_PATH = os.path.join(THIS_DIR, "resume_vectorizer.pkl")
SKILL_MAP_PATH = os.path.join(THIS_DIR, "skill_map.pkl")
SCORE_STATS_PATH = os.path.join(THIS_DIR, "score_stats.pkl")


# ------------------- Utilities -------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

STOPWORDS = {
    "and", "or", "the", "for", "with", "a", "an", "to", "of", "in", "on",
    "is", "are", "was", "were", "i", "you", "he", "she", "it", "they",
    "this", "that", "these", "those", "as", "by", "at", "from"
}

def extract_tokens(text: str):
    return [
        tok for tok in text.split()
        if len(tok) > 2 and tok not in STOPWORDS
    ]


# ------------------- Main Training -------------------

def main():
    print(f"Loading dataset from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Category", "Resume_str"])

    df["clean_text"] = df["Resume_str"].astype(str).apply(clean_text)

    # 1) Train classifier
    print("Training TF-IDF + Logistic Regression...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["Category"]

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X, y)

    joblib.dump(clf, FIELD_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Saved field_model.pkl + resume_vectorizer.pkl")

    # 2) Skill Map
    print("Building skill map per field...")

    field_skill_counts = defaultdict(Counter)

    for cat, text in zip(df["Category"], df["clean_text"]):
        tokens = extract_tokens(text)
        field_skill_counts[cat].update(tokens)

    TOP_N = 40
    skill_map = {
        field: [tok for tok, _ in counts.most_common(TOP_N)]
        for field, counts in field_skill_counts.items()
    }

    joblib.dump(skill_map, SKILL_MAP_PATH)
    print("Saved skill_map.pkl")

    # 3) Scoring statistics
    print("Creating scoring stats...")
    df["word_count"] = df["clean_text"].str.split().str.len()
    score_stats = {
        "mean_word_count": float(df["word_count"].mean()),
        "std_word_count": float(df["word_count"].std()) or 1.0,
    }

    joblib.dump(score_stats, SCORE_STATS_PATH)
    print("Saved score_stats.pkl")

    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
