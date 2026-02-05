# Step 1: Import libraries
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Step 2: Download NLTK data (only once)
nltk.download('stopwords')
nltk.download('punkt')

# Load CSV files
fake = pd.read_csv("news_full.csv")
true = pd.read_csv("news_clean_onlly.csv")


# Step 4: Add labels
fake["label"] = 0  # Fake news
true["label"] = 1  # Real news

# Step 5: Combine datasets
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)  # shuffle

# Step 6: Define stopwords
stop_words = set(stopwords.words('english'))

# Step 7: Clean text function
def clean_text(text):
    text = str(text)  # convert to string
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = word_tokenize(text)  # tokenize
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return ' '.join(words)

# Step 8: Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

# Step 9: Prepare features and labels
X = data['clean_text']  # text features
y = data['label']       # labels

# Step 10: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X)

# Step 11: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42
)

# Step 12: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 13: Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Step 14: Save the model and vectorizer for later use
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved! ✅")
