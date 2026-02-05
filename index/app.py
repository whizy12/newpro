# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import requests

# Load model & vectorizer
model = pickle.load(open("lr_model_balanced.pkl", "rb"))
vectorizer = pickle.load(open("lr_vectorizer_balanced.pkl", "rb"))

app = Flask(__name__)
CORS(app)

NEWS_API_KEY = "3358307798f84164adf991cbb1990a6f"  # Replace with your key
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_related_news(query, max_results=3):
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": max_results,
        "sortBy": "relevancy"
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        result = []
        for article in articles:
            result.append({
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "url": article.get("url"),
                "description": article.get("description")
            })
        return result
    return []

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X_vec = vectorizer.transform([text])
    proba = model.predict_proba(X_vec)[0]

    fake_prob = round(proba[0] * 100, 2)
    true_prob = round(proba[1] * 100, 2)

    prediction = "TRUE" if true_prob >= 40 else "FAKE"

    response_data = {
        "prediction": prediction,
        "true_probability": true_prob,
        "fake_probability": fake_prob,
        "related_news": []
    }

    if prediction == "TRUE":
        response_data["related_news"] = get_related_news(text[:80])

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
