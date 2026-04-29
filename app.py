from flask import Flask, request, jsonify
import joblib
from explain import explain_prediction

app = Flask(__name__)

# Load saved model files
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return "Spam Detection API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data["text"]

    # Convert text
    X = vectorizer.transform([text])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    label = "spam" if prediction == 1 else "not spam"

    # Explanation
    explanation = explain_prediction(text)

    return jsonify({
        "text": text,
        "prediction": label,
        "spam_probability": float(probability),
        "important_words": explanation
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)