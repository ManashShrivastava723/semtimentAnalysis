import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load your dataset
df = pd.read_csv('sentiment_dataset.csv')  # Replace with your dataset

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Splitting the data
X = df['clean_text']
y = df['label']  # Assuming 'label' is the column with sentiment labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Naive Bayes': MultinomialNB()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f'{name} Accuracy: {accuracy:.4f}')
    print(f'Classification Report for {name}:\n', classification_report(y_test, predictions))
    print(f'Confusion Matrix for {name}:\n', confusion_matrix(y_test, predictions))

# Selecting the best model based on performance
best_model = GradientBoostingClassifier(n_estimators=100)
best_model.fit(X_train_tfidf, y_train)
best_predictions = best_model.predict(X_test_tfidf)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, best_predictions)
print(f'Best Model (Gradient Boosting) Accuracy: {best_accuracy:.4f}')
print('Classification Report:\n', classification_report(y_test, best_predictions))
print('Confusion Matrix:\n', confusion_matrix(y_test, best_predictions))
