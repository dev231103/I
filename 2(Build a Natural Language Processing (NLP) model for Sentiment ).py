import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset


# Load IMDB dataset from Hugging Face
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
print(dataset)


# Convert dataset to pandas DataFrame
df = pd.DataFrame({
    "text": dataset["train"]["text"],
    "label": dataset["train"]["label"]
})


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Train logistic regression model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)


# Evaluate model
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Test with new sample text
new_text = ["I did not like this at all."]
new_vectorized = vectorizer.transform(new_text)
prediction = model.predict(new_vectorized)

print(
    f'Prediction for "{new_text[0]}": {"positive" if prediction[0] == 1 else "negative"}'
)
