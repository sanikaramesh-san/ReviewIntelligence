import pandas as pd
import string
import joblib
import nltk
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
print("Loading dataset...")
data = pd.read_csv("cleaned_reviews.csv")
print("Columns:", data.columns)
print("Dataset shape before cleaning:", data.shape)

# Remove rows where cleaned_review is empty or missing
data = data.dropna(subset=['cleaned_review'])
data = data[data['cleaned_review'].str.strip() != ""]

print("Dataset shape after removing empty reviews:", data.shape)


# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Clean reviews
#print("Cleaning reviews...")


# Save cleaned dataset
data.to_csv("cleaned_reviews.csv", index=False)

print("\nSample cleaned reviews:")
print(data[[ 'cleaned_review', 'sentiments']].head())

# Features and labels
X = data['cleaned_review']
y = data['sentiments']

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text to TF-IDF
print("Converting text to TF-IDF...")
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict
print("Making predictions...")
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n================ RESULTS ================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel saved to model/model.pkl")
print("Vectorizer saved to model/vectorizer.pkl")


# Prediction function
def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction


# Interactive testing
print("\n=============== TEST YOUR OWN REVIEW ===============")

while True:
    review = input("Enter a review (or type 'exit'): ")

    if review.lower() == 'exit':
        print("Goodbye!")
        break

    result = predict_sentiment(review)
    print("Predicted Sentiment:", result)
    print()