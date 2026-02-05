# test_model.py

import pickle

# Load trained model and vectorizer
model = pickle.load(open("lr_model_balanced.pkl", "rb"))
vectorizer = pickle.load(open("lr_vectorizer_balanced.pkl", "rb"))

def predict_news(text, threshold=0.4):
    X_vec = vectorizer.transform([text])
    proba = model.predict_proba(X_vec)[0]

    fake_prob = proba[0]
    true_prob = proba[1]

    label = "TRUE" if true_prob >= threshold else "FAKE"

    return {
        "prediction": label,
        "true_probability": round(true_prob * 100, 2),
        "fake_probability": round(fake_prob * 100, 2)
    }

# -------- TEST EXAMPLES --------
samples = [
    "The Federal Government announced new economic reforms on Tuesday according to the Ministry of Finance.",
    "Scientists confirm the earth will stop spinning tomorrow according to secret NASA documents.",
    "The Central Bank increased interest rates to curb inflation.",
    "Aliens spotted voting in Nigeria elections according to anonymous sources."
]

for text in samples:
    result = predict_news(text)
    print("\nText:", text)
    print("Prediction:", result["prediction"])
    print("True %:", result["true_probability"])
    print("Fake %:", result["fake_probability"])
