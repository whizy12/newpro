# preprocess_data.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)

# Load original dataset
df = pd.read_csv("news_full.csv")

# 🔥 FIX LABELS PROPERLY
df["Label"] = df["Label"].astype(str).str.strip().str.lower()

df["Label"] = df["Label"].replace({
    "fake": 0,
    "true": 1
})

# Drop rows that still have invalid labels
df = df[df["Label"].isin([0, 1])]
df["Label"] = df["Label"].astype(int)

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Feature engineering
df["char_count"] = df["clean_text"].apply(len)
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
df["avg_word_length"] = df["char_count"] / (df["word_count"] + 1)

# Remove empty text rows
df = df[df["word_count"] > 0]

# Save cleaned dataset
df.to_csv("news_clean_onlly.csv", index=False)

print("Preprocessing complete ✅")
print(df["Label"].value_counts())
