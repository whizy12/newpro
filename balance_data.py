# balance_data.py

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Load cleaned dataset
df = pd.read_csv("news_clean_onlly.csv")

# Separate features and label
X = df.drop(columns=["Label"])
y = df["Label"]

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combine back into one DataFrame
df_balanced = X_resampled.copy()
df_balanced["Label"] = y_resampled

# Shuffle rows
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Save balanced dataset
df_balanced.to_csv("news_balanced.csv", index=False)

print("Dataset balanced successfully ✅")
print(df_balanced["Label"].value_counts())
