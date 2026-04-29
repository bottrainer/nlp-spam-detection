import joblib
import numpy as np

# Load saved files
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def explain_prediction(text):
    # Transform input text
    X = vectorizer.transform([text])

    # Feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Model coefficients
    coefs = model.coef_[0]

    # Find words present in input
    indices = X.nonzero()[1]

    word_scores = {}

    for i in indices:
        word = feature_names[i]
        score = coefs[i]
        word_scores[word] = round(float(score), 4)

    # Top important words
    top_words = sorted(
        word_scores.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return top_words