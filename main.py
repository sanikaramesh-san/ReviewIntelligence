import joblib
import string

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Stopwords-free simple cleaner (same style as training)
def clean_text(text):
    text = str(text).lower()

    # keep "not" because it is important for sentiment
    important_words = ["not", "no", "never"]

    words = text.split()

    cleaned_words = []
    for word in words:
        word = ''.join(char for char in word if char not in string.punctuation)

        # keep negation words
        if word in important_words:
            cleaned_words.append(word)
        elif word not in [""]:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


def predict_sentiment(review):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction


print("\n===== SENTIMENT PREDICTION SYSTEM =====")

while True:
    review = input("\nEnter a review (or type 'exit'): ")

    if review.lower() == 'exit':
        print("Goodbye!")
        break

    result = predict_sentiment(review)
    print("Predicted Sentiment:", result)