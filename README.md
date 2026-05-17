# 🧠 Sentiment Analysis of Customer Reviews using Naive Bayes

## 📌 Problem Statement

E-commerce platforms receive a large number of customer reviews every day. Manually analyzing these reviews is time-consuming and inefficient.

The goal of this project is to build a Machine Learning model that automatically classifies customer reviews into:

- 😊 Positive  
- 😐 Neutral  
- 😞 Negative  

This helps businesses understand customer feedback and improve their products and services.

---

## 📂 Dataset

The dataset used in this project is:

- **File Name:** `cleaned_reviews.csv`  
- **Source:** Kaggle  
- **Type:** Preprocessed customer review dataset  

It contains customer reviews along with their corresponding sentiment labels.

---



## 🧠 Approach

The project follows a standard Natural Language Processing (NLP) pipeline:

### 1. Data Preprocessing
- Convert text to lowercase  
- Remove punctuation  
- Remove stopwords  

### 2. Feature Extraction
- TF-IDF (Term Frequency - Inverse Document Frequency) is used to convert text into numerical form  

### 3. Model Training
- Naive Bayes Classifier is used for sentiment classification  

### 4. Prediction
- The trained model predicts sentiment for new unseen reviews  

---

## 🤖 Machine Learning Algorithm

### ✔ Naive Bayes Classifier

Naive Bayes is a probabilistic classification algorithm based on Bayes’ Theorem.  
It is widely used for text classification problems because it works well with word frequencies.

---

## 📊 Model Workflow

1. Load dataset (`cleaned_reviews.csv`)  
2. Preprocess text data  
3. Convert text using TF-IDF vectorizer  
4. Train Naive Bayes model  
5. Evaluate accuracy  
6. Save model (`.pkl` files)  
7. Use `main.py` for predictions  

---

## 🧪 Example Predictions

| Review | Predicted Sentiment |
|--------|--------------------|
| "This product is amazing" | Positive 😊 |
| "it is worst" | Negative 😞 |
| "not bad" | Neutral 😐 |

---

## 📁 Files Description

- **sentiment_analysis.py** → Used to train the Naive Bayes model  
- **main.py** → Used to test and predict sentiment from user input  
- **cleaned_reviews.csv** → Dataset downloaded from Kaggle  
- **model folder** → Stores trained model and vectorizer  

---



## 🏁 Conclusion

This project successfully demonstrates how Machine Learning can be used to analyze customer sentiment from text data.  
It automates review classification and helps in better decision-making for businesses.

---

## 👨‍💻 Technologies Used

**Python**
Used as the main programming language for building the Machine Learning project because it is simple, powerful, and widely used for data science applications.
**Pandas**
Used for loading, reading, and handling the dataset (cleaned_reviews.csv) in a structured format for easy data manipulation and analysis.
**NumPy**
Used for performing numerical computations and supporting efficient processing of data in array format during model training.
**Scikit-learn**
Used to build and evaluate the Machine Learning model. It provides functions for splitting data, training the Naive Bayes classifier, and calculating accuracy.
**NLTK (Natural Language Toolkit)**
Used for Natural Language Processing tasks such as removing stopwords and cleaning text data before feeding it into the model.
**TF-IDF Vectorization**
Used to convert text data into numerical form so that the machine learning model can process it. It also highlights important words in the reviews.
**Naive Bayes Classifier**
A probabilistic Machine Learning algorithm used for sentiment classification. It is fast, efficient, and performs well for text-based problems like review analysis.
