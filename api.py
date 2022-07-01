from flask import Flask, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/evaluate", methods=["POST", "GET"])
def is_hate_speech():
    if request.method == "POST":
        max_length = 100
        trunc_type = "post"
        padding_type = "post"

        text = request.form["text"]
        model = tf.keras.models.load_model("hate_speech_model")
        sentence = [text]

        # loading tokenize
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(
            sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
        )
        score = f"{round(model.predict(padded)[0][0] * 100, 2)}%"
        data = {"text": text, "score": score}
        return data

    elif request.method == "GET":
        return "get?"
