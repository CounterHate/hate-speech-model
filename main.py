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
import pickle

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


def get_cleared_sentences(data):
    return [remove_emojis(d) for d in data]


def get_hate_speech(is_hate_speech=True, size=10000):
    url = "https://es.dc9.dev:9200/tweets/_search"
    headers = {"content-type": "application/json"}
    query = {
        "size": size,
        "query": {
            "function_score": {
                "query": {"match": {"is_hate_speech": is_hate_speech}},
                "functions": [{"random_score": {}}],
            }
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
    raw_data = get_hate_speech(is_hate_speech=False, size=10_000)
    data = []
    for item in raw_data:
        if "is_hate_speech" not in item["_source"]:
            continue
        data.append(item["_source"])

    raw_data = get_hate_speech(is_hate_speech=True, size=10_000)
    for item in raw_data:
        if "is_hate_speech" not in item["_source"]:
            continue
        data.append(item["_source"])

    sentences = []
    labels = []

    for item in data:
        if item["is_hate_speech"] is True or item["is_hate_speech"] is False:
            sentences.append(item["content"])
            labels.append(item["is_hate_speech"])
        else:
            print(item["is_hate_speech"])
    sentences = get_cleared_sentences(sentences)

    mid_data_point = math.floor(len(sentences) * 0.8)
    training_sentences = sentences[0:mid_data_point]
    testing_sentences = sentences[mid_data_point:]
    training_labels = labels[0:mid_data_point]
    testing_labels = labels[mid_data_point:]

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
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        run_eagerly=True,
    )

    model.summary()

    num_epochs = 30
    history = model.fit(
        training_padded,
        training_labels,
        epochs=num_epochs,
        validation_data=(testing_padded, testing_labels),
        verbose=2,
    )
    model.evaluate(testing_padded, testing_labels)
    model.save("hate_speech_model")

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # sentence = ["Wypierdalaj do siebie pejsata mendo", "Lubię pierogi"]
    # sequences = tokenizer.texts_to_sequences(sentence)
    # padded = pad_sequences(
    #     sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    # )
    # print(model.predict(padded))


if __name__ == "__main__":
    main()
