from keras.datasets import reuters
from keras.utils import to_categorical
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) \
    = reuters.load_data(num_words=10000)

# word_index = reuters.get_word_index()
# reverse_word_index = dict(
#     [(value, key) for (key, value) in word_index.items()])
# decoded = ' '.join([reverse_word_index.get(i-3, '?') 
#                     for i in train_data[1]])
# 
# print(decoded)

def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorise_sequences(train_data)
x_test = vectorise_sequences(test_data)

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


# the last example was a binary classiffication but now
# there's 46 classes. each layer can only get information
# form the layer behind it, making each layer a potetial
# bottleneck of the model. for this reason increase units.
model = models.Sequential()
model.add(layers.Dense(
    64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# softmax to return probability distribution
model.add(layers.Dense(46, activation='softmax'))

# crossentropy will compare the distribution of the model
# and true values
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()