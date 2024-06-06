from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer  # Data transformation
from sklearn.model_selection import train_test_split  # Data testing
from sklearn.linear_model import LogisticRegression  # Prediction Model
import pickle

# Comparison between real and predicted
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import (
    LabelEncoder,
)  # Variable encoding and decoding for XGBoost
import re  # Regular expressions
import nltk
from nltk import word_tokenize
from pathlib import Path
import numpy as np
from collections import Counter
from pathlib import Path

nltk.download("stopwords")
nltk.download("punkt")

# Znalezienie root projectu
current_path = Path().cwd()
root_project_path = current_path.parent.parent  # Przechodzimy dwa poziomy w górę
datasets_path = root_project_path / 'datasets'

# Construct the path to the CSV file
csv_path_train = datasets_path / "twitter_training.csv"
csv_path_val = datasets_path / "twitter_validation.csv"

# Read the CSV file
train = pd.read_csv(csv_path_train, encoding='utf-8', header=None)
val = pd.read_csv(csv_path_val, encoding='utf-8', header=None)

train_data = train

val_data = val

train_data = train_data.dropna()
val_data = val_data.dropna()

# Text transformation
train_data["lower"] = train_data.text.str.lower()  # lowercase
# converting all to string
train_data["lower"] = [str(data) for data in train_data.lower]
train_data["lower"] = train_data.lower.apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))  # regex

val_data["lower"] = val_data.text.str.lower()  # lowercase
val_data["lower"] = [str(data)
                     for data in val_data.lower]  # converting all to string
val_data["lower"] = val_data.lower.apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))  # regex

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')

reviews_train, reviews_test = train_test_split(
    train_data, test_size=0.2, random_state=43)

bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words,
    ngram_range=(1, 1)
)

# Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
# Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)

# Labels for train and test encoding
y_train_bow = reviews_train['sentiment']
y_test_bow = reviews_test['sentiment']

# Logistic regression
model1 = LogisticRegression(C=1, solver="liblinear", max_iter=150)
model1.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model1.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)

# Validation data
X_val_bow = bow_counts.transform(val_data.lower)
y_val_bow = val_data['sentiment']

val_res = model1.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, val_res) * 100)

# n-gram of 4 words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1, 4)
)
# Data labeling
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

model2 = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
# Logistic regression
model2.fit(X_train_bow, y_train_bow)
# Prediction
test_pred_2 = model2.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred_2) * 100)

y_val_bow = val_data['sentiment']
val_pred_2 = model2.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, val_pred_2) * 100)

# Assuming current_path is defined somewhere in your code
root_project_path = current_path.parent.parent  # Go up two levels
datasets_path = root_project_path / 'models'

# Check if the directory exists, if not create it
if not datasets_path.exists():
    datasets_path.mkdir(parents=True, exist_ok=True)

# Define the filename and save the model
filename = datasets_path / 'sentimaniac.pkl'
pickle.dump(model2, open(filename, 'wb'))

pickled_model = pickle.load(open('finalized_model.pkl', 'rb'))
Val_pred_2 = pickled_model.predict(X_val_bow)
