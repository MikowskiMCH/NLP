import numpy as np
import tensorflow as tf
from tensorflow import keras
import string
import re



batch_size = 32
max_features = 20000
max_length = 500
embedding_dim = 128

x_train = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    validation_split=0.2,
    subset='training',
    seed=1337,
    batch_size=batch_size,
)
x_val = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    validation_split=0.2,
    subset='validation',
    seed=1337,
    batch_size=batch_size,
)
x_test = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=batch_size,
)


def custom_standardize(input_data):
    lowercase=tf.strings.lower(input_data)
    strip = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(strip, f'[{re.escape(string.punctuation)}]', '')


vectorize = tf.keras.layers.TextVectorization(
    standardize = custom_standardize,
    output_mode='int',
    max_tokens=max_features,
    output_sequence_length=max_length,
)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize(text), label


text_ds = x_train.map(lambda x, labels: x)
vectorize.adapt(text_ds)


train_ds = x_train.map(vectorize_text)
val_ds = x_val.map(vectorize_text)
test_ds = x_test.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = keras.Sequential([
    keras.layers.Embedding(max_features, embedding_dim),
    keras.layers.Dropout(0.5),
    keras.layers.Conv1D(embedding_dim,3,activation='relu'),
    keras.layers.GlobalMaxPooling1D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation="relu"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds,validation_data=val_ds,epochs=3)
model.evaluate(test_ds)