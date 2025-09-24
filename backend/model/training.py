from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

# Load the datasets
df_fake = pd.read_csv("C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\dataset\\Fake.csv")
df_true = pd.read_csv("C:\\Users\\amrit\\Documents\\GitHub\\NEWS_DETECTION\\dataset\\True.csv")

# Add labels: 0 = Fake, 1 = True
df_fake["label"] = 0
df_true["label"] = 1

# Combine both
df = pd.concat([df_fake, df_true], ignore_index=True)

# Optional: Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
x = df["text"]
y = df["label"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Model training
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# Create output folder
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")