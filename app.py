import numpy as np
import pandas as pd
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import joblib

# -----------------------------
# Load Dataset
# -----------------------------

fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

df = pd.concat([fake_df, true_df], axis=0)

df = df[['title', 'text', 'label']]

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\\S+', '', text)

    text = re.sub(r'\\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\\s+', ' ', text).strip()

    return text

df['cleaned_text'] = df['title'] + ' ' + df['text']

df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# -----------------------------
# Encode Labels
# -----------------------------

df['label_num'] = df['label'].map({
    'REAL': 0,
    'FAKE': 1
})

# -----------------------------
# Split Dataset
# -----------------------------

X = df['cleaned_text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Models
# -----------------------------

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    print(f'\\nTraining {name}...')

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(f'Accuracy : {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1 Score : {f1:.4f}')

# -----------------------------
# Best Model
# -----------------------------

best_model_name = max(results, key=lambda x: results[x]['f1'])

best_model = results[best_model_name]['model']

print(f'\\nBest Model: {best_model_name}')

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(best_model, 'models/fake_news_model.pkl')

joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print('\\nModel saved successfully!')

# -----------------------------
# Custom Prediction
# -----------------------------

def predict_news(news):

    cleaned = clean_text(news)

    vectorized = vectorizer.transform([cleaned])

    prediction = best_model.predict(vectorized)[0]

    if prediction == 1:
        print('\\nPrediction: FAKE NEWS')
    else:
        print('\\nPrediction: REAL NEWS')

predict_news(
    'NASA confirms aliens landed in New York'
)

predict_news(
    'Government announces new education policy'
)
