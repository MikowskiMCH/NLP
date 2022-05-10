import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt

dataset, info = tfds.load(name="imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = dataset['train'], dataset['test']
train_data = train_data.shuffle(10000).batch(50).prefetch(tf.data.AUTOTUNE)
train_data = train_data.cache()
test_data = test_data.batch(50).prefetch(tf.data.AUTOTUNE)
test_data = test_data.cache()

encoder = TextVectorization(max_tokens=20000)
encoder.adapt(train_data.map(lambda x, y: x))

model = tf.keras.Sequential()
model.add(encoder)
model.add(tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=8))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data, epochs=40, validation_data=test_data)
model.summary()


loss, accuracy = model.evaluate(test_data)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Strata treningowa')
plt.plot(epochs, val_loss, 'b', label='Strata walidacyjna')
plt.title('Strata treningowa i walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Dokładność treningowa')
plt.plot(epochs, val_acc, 'b', label='Dokładność walidacyjna')
plt.title('Dokładność treningowa i walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend(loc='lower right')
plt.show()
model.save('ModelTestowy')


