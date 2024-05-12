import re  # Regular expressions
import warnings
from pathlib import Path

import matplotlib.pyplot as plt  # Plotting properties
import nltk
import pandas as pd
import seaborn as sns  # Plotting properties
from nltk import word_tokenize
from sklearn.feature_extraction.text import \
    CountVectorizer  # Data transformation
from sklearn.linear_model import LogisticRegression  # Prediction Model
# Comparison between real and predicted
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split  # Data testing
from sklearn.preprocessing import \
    LabelEncoder  # Variable encoding and decoding for XGBoost
from wordcloud import WordCloud  # Word visualization
from xgboost import XGBClassifier

nltk.download("stopwords")
nltk.download("punkt")
# Construct the path to the CSV file
csv_path_train = Path("datasets") / Path("twitter_training.csv")
csv_path_val = Path("datasets") / Path("twitter_validation.csv")

# Read the CSV file
train = pd.read_csv(csv_path_train, encoding="utf-8", header=None)
val = pd.read_csv(csv_path_val, encoding="utf-8", header=None)

train.columns = ["id", "type", "sentiment", "text"]
train.head()

val.columns = ["id", "type", "sentiment", "text"]
val.head()

train_data = train
train_data

val_data = val
val_data

train.sentiment.value_counts()

sentiment_counts = train["sentiment"].value_counts()

labels = ["Positive", "Neutral", "Negative", "Irrelevant"]
colors = ["green", "gray", "red", "orange"]

plt.figure(figsize=(8, 6))
plt.bar(labels, sentiment_counts, color=colors)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution for train set")
plt.show()

val.sentiment.value_counts()

sentiment_counts = val["sentiment"].value_counts()

labels = ["Positive", "Neutral", "Negative", "Irrelevant"]
colors = ["green", "gray", "red", "orange"]

plt.figure(figsize=(8, 6))
plt.bar(labels, sentiment_counts, color=colors)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution for val set")
plt.show()

train_data.isna().sum().sum()
train_data = train_data.dropna()
train_data.isna().sum().sum()
train_data
val_data.isna().sum().sum()
val_data = val_data.dropna()
val_data.isna().sum().sum()

# Text transformation
train_data["lower"] = train_data.text.str.lower()  # lowercase
train_data["lower"] = [
    str(data) for data in train_data.lower
]  # converting all to string
train_data["lower"] = train_data.lower.apply(
    lambda x: re.sub("[^A-Za-z0-9 ]+", " ", x)
)  # regex
val_data["lower"] = val_data.text.str.lower()  # lowercase
val_data["lower"] = [str(data) for data in val_data.lower]  # converting all to string
val_data["lower"] = val_data.lower.apply(
    lambda x: re.sub("[^A-Za-z0-9 ]+", " ", x)
)  # regex

train_data.head()

# positive
word_cloud_text = "".join(train_data[train_data["sentiment"] == "Positive"].lower)
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
word_cloud_text = "".join(train_data[train_data["sentiment"] == "Negative"].lower)
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
word_cloud_text = "".join(train_data[train_data["sentiment"] == "Neutral"].lower)
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

# irrelevant
word_cloud_text = "".join(train_data[train_data["sentiment"] == "Irrelevant"].lower)
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

# Count information per category
plot1 = train_data.groupby(by=["type", "sentiment"]).count().reset_index()
plot1.head()

# Figure of comparison per branch
plt.figure(figsize=(20, 6))
sns.barplot(data=plot1, x="type", y="id", hue="sentiment")
plt.xticks(rotation=90)
plt.xlabel("Brand")
plt.ylabel("Number of tweets")
plt.grid()
plt.title("Distribution of tweets per Branch and Type")

# Text splitting
tokens_text = [word_tokenize(str(word)) for word in train_data.lower]
# Unique word counter
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Number of tokens: ", len(set(tokens_counter)))

tokens_text[23]

stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words("english")
stop_words[:10]

bow_counts = CountVectorizer(
    tokenizer=word_tokenize, stop_words=stop_words, ngram_range=(1, 1)
)

reviews_train, reviews_test = train_test_split(
    train_data, test_size=0.2, random_state=43
)

# Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
# Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)

X_test_bow

# Labels for train and test encoding
y_train_bow = reviews_train["sentiment"]
y_test_bow = reviews_test["sentiment"]

# Total of registers per category
y_test_bow.value_counts() / y_test_bow.shape[0]

# Logistic regression
model1 = LogisticRegression(C=1, solver="liblinear", max_iter=150)
model1.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model1.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)

print("Classification Report for Test Data:")
print(classification_report(y_test_bow, test_pred))

import numpy as np

# Generate Classification Report
report = classification_report(y_test_bow, test_pred, output_dict=True)

# Extract metrics for each class
classes = list(report.keys())[
    :-3
]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
precision = [report[class_]["precision"] for class_ in classes]
recall = [report[class_]["recall"] for class_ in classes]
f1_score = [report[class_]["f1-score"] for class_ in classes]

# Plotting
x = np.arange(len(classes))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
precision_bar = ax.bar(x - width, precision, width, label="Precision")
recall_bar = ax.bar(x, recall, width, label="Recall")
f1_score_bar = ax.bar(x + width, f1_score, width, label="F1-Score")

# Add labels, title, and legend
ax.set_xlabel("Sentiment Class")
ax.set_ylabel("Score")
ax.set_title("Classification Report Metrics by Sentiment Class")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()


# Add value labels on top of the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            "{}".format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


add_value_labels(precision_bar)
add_value_labels(recall_bar)
add_value_labels(f1_score_bar)

plt.tight_layout()
plt.show()


# Validation data
X_val_bow = bow_counts.transform(val_data.lower)
y_val_bow = val_data["sentiment"]

X_val_bow

val_res = model1.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, val_res) * 100)

# n-gram of 4 words
bow_counts = CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 4))
# Data labeling
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

X_train_bow

model2 = LogisticRegression(C=0.9, solver="liblinear", max_iter=1500)
# Logistic regression
model2.fit(X_train_bow, y_train_bow)
# Prediction
test_pred_2 = model2.predict(X_test_bow)
print("Accuracy: ", accuracy_score(y_test_bow, test_pred_2) * 100)

y_val_bow = val_data["sentiment"]
Val_pred_2 = model2.predict(X_val_bow)
print("Accuracy: ", accuracy_score(y_val_bow, Val_pred_2) * 100)


def predict_sentiment(model, bow_counts, sentence):
    # Preprocess input sentence
    processed_sentence = re.sub("[^A-Za-z0-9 ]+", " ", sentence.lower())
    # print(processed_sentence)
    # Transform sentence using the same CountVectorizer
    transformed_sentence = bow_counts.transform([processed_sentence])
    # print(transformed_sentence)
    # Predict sentiment
    prediction = model2.predict(transformed_sentence)
    # print(prediction)
    if prediction == "Positive":
        return "Positive"
    elif prediction == "Neutral":
        return "Neutral"
    elif prediction == "Negative":
        return "Negative"
    else:
        return "Irrelevant"


test_sentence = input("Sentence: ")
print(
    f"Sentiment for '{test_sentence}':",
    predict_sentiment(model2, bow_counts, test_sentence),
)
