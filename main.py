import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras import layers
from keras.layers import TextVectorization
import string
import re

batch_size = 32
max_features = 20000
max_length = 150
embedding_dim = 128

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    batch_size=batch_size,
    shuffle_files=True,
    split=('train[:80%]', 'train[20%:]', 'test'),
    as_supervised=True)



def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_length
)


text_ds = train_data.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)


def vectorize_text(text, label):
    return vectorize_layer(text), label


train_ds = train_data.map(vectorize_text)
validation_ds = validation_data.map(vectorize_text)
test_ds = test_data.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)


inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Conv1D(128, 15, padding='valid', activation='relu', strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(1, activation='relu')(x)

model = keras.Model(inputs, predictions)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_ds, validation_data=validation_ds, epochs=5)
model.evaluate(test_ds)
