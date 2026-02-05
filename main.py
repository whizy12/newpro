import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("news_clean_onlly.csv")

X_test = df["text"]
y_test = df["label"]

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ss