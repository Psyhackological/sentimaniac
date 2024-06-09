import re
import pickle


def predict_sentiment(pipeline, sentence):
    # Preprocess input sentence
    processed_sentence = re.sub("[^A-Za-z0-9 ]+", " ", sentence.lower())

    # Transform and predict sentiment
    transformed_sentence = pipeline["vectorizer"].transform(
        [processed_sentence])
    prediction = pipeline["model"].predict(transformed_sentence)

    if prediction == "Positive":
        return "Positive"
    elif prediction == "Neutral":
        return "Neutral"
    elif prediction == "Negative":
        return "Negative"
    else:
        return "Irrelevant"


# Load the pipeline (model and vectorizer)
pipeline = pickle.load(open('sentimaniac.pkl', 'rb'))

test_sentence = input("Sentence: ")
print(
    f"Sentiment for '{test_sentence}':",
    predict_sentiment(pipeline, test_sentence),
)
