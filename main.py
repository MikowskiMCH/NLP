import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

model = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
hub_layer = hub.KerasLayer(model, output_shape=[128], input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Reshape(target_shape=[1, 128]))
model.add(tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

x_val = train_examples[:10000]
partial_x_train = train_examples[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=3,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    verbose=1)

import numpy as np
import torch

from textattack.models.wrappers import ModelWrapper

class CustomTensorFlowModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, text_input_list):
        text_array = np.array(text_input_list)
        preds = self.model(text_array).numpy()
        logits = torch.exp(-torch.tensor(preds))
        logits = 1 / (1 + logits)
        logits = logits.squeeze(dim=-1)
        # Since this model only has a single output (between 0 or 1),
        # we have to add the second dimension.
        final_preds = torch.stack((1-logits, logits), dim=1)
        return final_preds


model_wrapper = CustomTensorFlowModelWrapper(model)

from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import PWWSRen2019
from textattack import Attacker
from textattack import AttackArgs

dataset = HuggingFaceDataset("rotten_tomatoes", None, "test", shuffle=True)
attack = PWWSRen2019.build(model_wrapper)
attack_args = AttackArgs(num_examples=100, checkpoint_dir="checkpoints")
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()