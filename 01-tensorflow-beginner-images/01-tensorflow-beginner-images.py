'''
this file has been inspired by the following article:
https://www.tensorflow.org/tutorials/keras/basic_classification
'''

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# dataset with 60000 labeled images of fashion objects
# each image is 28x28 with pixel values from 0 to 255
fashion_mnist = keras.datasets.fashion_mnist

# the labels are an array from 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# all the variables assigned here are numpy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

##--- Test and show the dataset information
print("There are %d training images" % (len(train_labels)))
print("There are %d test images" % (len(test_labels)))

plt.figure()
plt.imshow(train_images[0])
plt.xlabel("Unmodified image from the dataset")
plt.colorbar()
plt.grid(False)
plt.show()

# scale the image values to a range of 0 to 1
# this has to be done for both the training and test dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# display the first 25 images with their labels
print("Now showing the first image of the dataset just to show you and waste CPU time...")
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

##--- Build the neural network model
# setup the layers for the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # image of 28x28 pixels
    keras.layers.Dense(128, activation=tf.nn.relu), # 128 neurons
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 neurons, each with probabiliy scores that sum to 1
])

# compile the model
model.compile(
    optimizer='adam', # strategy of model update during training
    loss='sparse_categorical_crossentropy', # to measure how accurate the model is
    metrics=['accuracy'] # how we monitor the training and testing
)

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy using the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# predict the test images
# this is an array with 10 numbers that describe the "confidence" of the model for each clothing class
predictions = model.predict(test_images)

# some helper functions to show images
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# show the 0th image of the test dataset
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# plot the first X test images, their predicted label, and the true label
# color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
