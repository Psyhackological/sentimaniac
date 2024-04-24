# Variable encoding and decoding for XGBoost
import re  # Regular expressions
import warnings
from pathlib import Path

import matplotlib.pyplot as plt  # Plotting properties
import nltk
import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer  # Data transformation
from sklearn.linear_model import LogisticRegression  # Prediction Model

# Comparison between real and predicted
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # Data testing
from wordcloud import WordCloud  # Word visualization

nltk.download("stopwords")
nltk.download("punkt")

# Construct the path to the CSV file
csv_path_train = Path("datasets") / Path("Twitter_Data.csv")
csv_path_test = Path("datasets") / Path("Reddit_Data.csv")

# Read the CSV file
train = pd.read_csv(csv_path_train, encoding="iso-8859-1")
test = pd.read_csv(csv_path_test, encoding="iso-8859-1")

train.columns = ["text", "Sentiment"]
test.columns = ["text", "Sentiment"]

train = train[["text", "Sentiment"]]
test = test[["text", "Sentiment"]]

train.Sentiment.value_counts()

test.Sentiment.value_counts()

train.isna().sum().sum()
train = train.dropna()
train.isna().sum().sum()
test.isna().sum().sum()
test = test.dropna()
test.isna().sum().sum()

train_data = train
train_data

test_data = test
test_data


# Text transformation
train_data["lower"] = train_data.text.str.lower()  # lowercase
train_data["lower"] = [
    str(data) for data in train_data.lower
]  # converting all to string
train_data["lower"] = train_data.lower.apply(
    lambda x: re.sub("[^A-Za-z0-9 ]+", " ", x),
)  # regex
test_data["lower"] = test_data.text.str.lower()  # lowercase
test_data["lower"] = [str(data) for data in test_data.lower]  # converting all to string
test_data["lower"] = test_data.lower.apply(
    lambda x: re.sub("[^A-Za-z0-9 ]+", " ", x),
)  # regex

train_data.head()

# positive
word_cloud_text = "".join(train_data[train_data["Sentiment"] == 1].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800,
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# neutral
word_cloud_text = "".join(train_data[train_data["Sentiment"] == 0].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800,
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# negative
word_cloud_text = "".join(train_data[train_data["Sentiment"] == -1].lower)
# Creation of wordcloud
wordcloud = WordCloud(
    max_font_size=100,
    max_words=100,
    background_color="black",
    scale=10,
    width=800,
    height=800,
).generate(word_cloud_text)
# Figure properties
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Text splitting
tokens_text = [word_tokenize(str(word)) for word in train_data.lower]
# Unique word counter
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))

tokens_text[1]

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words("english")
stop_words[:5]

bow_counts = CountVectorizer(
    tokenizer=word_tokenize, stop_words=stop_words, ngram_range=(1, 1),
)

reviews_train, reviews_test = train_test_split(
    train_data, test_size=0.2, random_state=43,
)

warnings.filterwarnings(
    "ignore",
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
)
warnings.filterwarnings(
    "ignore", message="Your stop_words may be inconsistent with your preprocessing",
)

# Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
# Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)

X_test_bow

# Labels for train and test encoding
y_train_bow = reviews_train["Sentiment"]
y_test_bow = reviews_test["Sentiment"]

# Total of registers per category
y_test_bow.value_counts() / y_test_bow.shape[0]

# Logistic regression
model1 = LogisticRegression(C=1, solver="liblinear", max_iter=150)
model1.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model1.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)

# Validation data
X_val_bow = bow_counts.transform(test_data.lower)
y_val_bow = test_data["Sentiment"]

X_val_bow

val_res = model1.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, val_res) * 100)


def predict_sentiment(model, bow_counts, sentence):
    # Preprocess input sentence
    processed_sentence = re.sub("[^A-Za-z0-9 ]+", " ", sentence.lower())
    processed_sentence
    # Transform sentence using the same CountVectorizer
    transformed_sentence = bow_counts.transform([processed_sentence])
    transformed_sentence
    # Predict sentiment
    prediction = model1.predict(transformed_sentence)
    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Neutral"
    else:
        return "Negative"


test_sentence = "Building meaningful relationships based on trust and mutual respect leads to a happier life."
print(
    f"Sentiment for '{test_sentence}':",
    predict_sentiment(model1, bow_counts, test_sentence),
)
