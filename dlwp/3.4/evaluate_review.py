from keras.datasets import imdb
from keras import models
import os
import numpy as np

# from listing import vectorise_sequences


def vectorise_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

print("Type review:")
review = input()

# encode review
word_index = imdb.get_word_index()
words = review.split()
indices = [word_index[s] for s in words]
print(indices)

model_in = vectorise_sequences([indices])

path = os.path.join(os.getcwd(), 'imdb_review.keras')
model = models.load_model(path)

print(model_in.shape)
print(model.predict(model_in))