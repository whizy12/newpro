# train_logistic_balanced.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load balanced dataset
df = pd.read_csv("news_balanced.csv")

# Features and label
X = df["clean_text"]
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression (balanced is now optional but safe)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["FAKE", "TRUE"]))

# Save model and vectorizer
with open("lr_model_balanced.pkl", "wb") as f:
    pickle.dump(model, f)

with open("lr_vectorizer_balanced.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nBalanced Logistic Regression model saved ✅")
