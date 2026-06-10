
import os
import warnings

import joblib
from flask import Flask, jsonify, render_template, request
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load("spam_stacking_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as exc:
    model = None
    vectorizer = None
    print("Warning: could not load model files:", exc)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["message"]

    # Convert to lowercase
    text_lower = text.lower()

    # Spam keyword list
    spam_keywords = [
        "loan", "eligible", "winner", "won", "gift",
        "free", "click here", "claim", "offer",
        "urgent", "credit", "cash", "prize",
        "reward", "bonus", "congratulations",
        "limited time", "bank offer","now", "exclusive", "act now", "don't miss"
    ]

    # Rule-based detection first
    if any(word in text_lower for word in spam_keywords):
        result = "Spam"

    else:
        # ML Prediction (with a safe fallback if the saved estimator is incompatible)
        if model is not None and vectorizer is not None:
            try:
                vec = vectorizer.transform([text])
                pred = model.predict(vec)[0]
                result = "Spam" if pred == 1 else "Ham"
            except Exception:
                result = "Ham"
        else:
            result = "Ham"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

