import requests
from requests.auth import HTTPBasicAuth
import json
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import math


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
training_size = 20000


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


def remove_emojis(data):
    emoj = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, "", data)


def decode_sentence(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def get_cleared_sentences(data):
    return [remove_emojis(d) for d in data]


def get_data(test=False, size=10_000):
    filter_query = dict()
    if test:
        filter_query["bool"] = {
            "must": [
                {
                    "match": {
                        "lang": "pl",
                    },
                },
                {
                    "match": {
                        "is_retweet": False,
                    },
                },
            ],
            "must_not": [
                {
                    "match": {
                        "is_hate_speech": True,
                    },
                },
                {
                    "match": {
                        "is_hate_speech": False,
                    },
                },
            ],
        }
    else:
        filter_query["bool"] = {
            "must": [
                {
                    "match": {
                        "lang": "pl",
                    },
                },
                {
                    "match": {
                        "is_retweet": False,
                    },
                },
            ],
            "should": [
                {
                    "match": {
                        "is_hate_speech": True,
                    },
                },
                {
                    "match": {
                        "is_hate_speech": False,
                    },
                },
            ],
        }

    url = "https://es.dc9.dev:9200/tweets/_search"
    headers = {"content-type": "application/json"}
    query = {
        "size": size,
        "_source": ["content", "is_hate_speech"],
        "query": {
            "function_score": {"random_score": {}, "query": filter_query},
        },
    }

    r = requests.get(
        url=url,
        data=json.dumps(query),
        headers=headers,
        auth=HTTPBasicAuth("dc9", "hohC2wix"),
    )
    return r.json()["hits"]["hits"]


def main():
    raw_data = get_data(test=False, size=10_000)
    data = []

    for item in raw_data:
        if "is_hate_speech" not in item["_source"]:
            continue

    mid_data_point = math.floor(len(data) / 2)
    training_sentences = []
    testing_sentences = []
    training_labels = []
    testing_labels = []

    for item in data[0:mid_data_point]:
        if (
            item["_source"]["is_hate_speech"] is True
            or item["_source"]["is_hate_speech"] is False
        ):
            training_sentences.append(item["_source"]["content"])
            training_labels.append(item["_source"]["is_hate_speech"])
    training_sentences = get_cleared_sentences(training_sentences)

    for item in data[mid_data_point:]:
        if (
            item["_source"]["is_hate_speech"] is True
            or item["_source"]["is_hate_speech"] is False
        ):
            testing_sentences.append(item["_source"]["content"])
            testing_labels.append(item["_source"]["is_hate_speech"])
    testing_sentences = get_cleared_sentences(training_sentences)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(
        training_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type,
    )

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(
        testing_sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type,
    )

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, input_length=max_length
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True)

    model.summary()

    num_epochs = 30
    history = model.fit(
        training_padded,
        training_labels,
        epochs=num_epochs,
        validation_data=(testing_padded, testing_labels),
        verbose=2,
    )

    return

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    print(decode_sentence(training_padded[0]))
    print(training_sentences[2])
    print(training_labels[2])


if __name__ == "__main__":
    main()
