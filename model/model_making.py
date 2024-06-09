from pathlib import Path
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

nltk.download("stopwords")
nltk.download("punkt")

# Znalezienie root projectu
current_path = Path().cwd()
root_project_path = current_path.parent  # Przechodzimy dwa poziomy w górę
datasets_path = root_project_path / 'datasets'

# Construct the path to the CSV file
csv_path_train = datasets_path / "twitter_training.csv"
csv_path_val = datasets_path / "twitter_validation.csv"

# Read the CSV file
train = pd.read_csv(csv_path_train, encoding='utf-8', header=None)
val = pd.read_csv(csv_path_val, encoding='utf-8', header=None)

train.columns = ['id', 'type', 'sentiment', 'text']
val.columns = ['id', 'type', 'sentiment', 'text']

train_data = train.dropna()
val_data = val.dropna()

# Text transformation
train_data["lower"] = train_data.text.str.lower().apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', str(x)))  # lowercase and regex

val_data["lower"] = val_data.text.str.lower().apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', str(x)))  # lowercase and regex

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')

reviews_train, reviews_test = train_test_split(
    train_data, test_size=0.2, random_state=43)

bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words,
    ngram_range=(1, 4)  # n-gram of 4 words
)

# Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

# Labels for train and test encoding
y_train_bow = reviews_train['sentiment']
y_test_bow = reviews_test['sentiment']

model = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
# Logistic regression
model.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model.predict(X_test_bow)
print("Test Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)

val_pred = model.predict(X_val_bow)
print("Validation Accuracy: ", accuracy_score(
    val_data['sentiment'], val_pred) * 100)

# Zapisz model i CountVectorizer razem
pipeline = {
    "model": model,
    "vectorizer": bow_counts
}

# Assuming current_path is defined somewhere in your code
root_project_path = current_path.parent  # Go up two levels
model_path = root_project_path / 'model'

# Check if the directory exists, if not create it
if not model_path.exists():
    model_path.mkdir(parents=True, exist_ok=True)

# Define the filename and save the model
filename = model_path / 'sentimaniac.pkl'
with open(filename, 'wb') as f:
    pickle.dump(pipeline, f)
