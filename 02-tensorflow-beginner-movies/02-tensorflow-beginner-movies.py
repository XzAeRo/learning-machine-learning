'''
this file has been inspired by the following article:
https://www.tensorflow.org/tutorials/keras/basic_text_classification
'''

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# the size of the vocabulary we will use from the reviews
vocab_size = 10000

# dataset with 25000/25000 train/test imdb reviews preoprocessed into integers of a dictionary
# in this case we import the dataset with the top 10000 words used in reviews, the rest is discarded
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# the dictionary which the datasets use to preprocess the reviews
word_index = imdb.get_word_index()

# a helper function to decode reviews back into text

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(colors.BOLD + colors.GREEN + "Now printing an example review from training data..." + colors.END)
print(decode_review(train_data[0]))

# PREPARE THE DATA
# since the reviews have variable lenghts, we will pad them to make them all equal
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(colors.BOLD + colors.GREEN + "Now printing the same review but with padded data" + colors.END)
print(decode_review(train_data[0]))

# BUILD THE MODEL
# a neural network need layers and neurons
# how many layer and neurons is THE architectural decision when building the model
model = keras.Sequential() # a sequential layer system
model.add(keras.layers.Embedding(vocab_size, 16)) # first layer: looks up the embedding vector for each word-index. The vectors are learned as the model trains.
model.add(keras.layers.GlobalAveragePooling1D()) # second layer: fixed-length output for each example by averaging the sequence dimension. Allows to handle variable length input.
model.add(keras.layers.Dense(16, activation=tf.nn.relu)) # third layer: 16 neurons which are fed with the previous layer in a fully conected layer design (dense)
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) # last layer: single-output neuron with the sigmoid activation function (values 0 to 1 indicating confidence levels)

print(model.summary())

# compìle the model
model.compile(optimizer='adam',
              loss='binary_crossentropy', # we could MSE, but binary crossentropy is better with probabilities
              metrics=['acc']) # we will monitor accuracy to train our model

# create a validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# wait, why not use the test data?
# A: we want to use our test data only to evaluate our accuracy...
# we should only use our training data to develop and tune our model

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# let's evaluate our model
results = model.evaluate(test_data, test_labels)
print("%s%sOur models has an accuracy of %.3f and a loss of %.3f%s" % (colors.BOLD, colors.GREEN, results[1], results[0], colors.END))

# let's see graphically what happened over time
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# we conclude that after 20 or so epochs, the models starts overfitting
# this means that our model gets over-adapted to the training data
# and performs better on training data than on data that has never seen before