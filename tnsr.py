from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.preprocessing.image import ImageDataGenerator



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plot.grid(False)
  plot.xticks([])
  plot.yticks([])

  plot.imshow(img, cmap=plot.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plot.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plot.grid(False)
  plot.xticks([])
  plot.yticks([])
  thisplot = plot.bar(range(10), predictions_array, color="#777777")
  plot.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')






print(tf.__version__)

dataset = keras.datasets.cifar10
#(x_train, y_train),           (x_test, y_test)
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

"""
print(train_images.shape)
print(len(train_labels))
test_images.shape
len(test_labels)
"""




"""
plot.figure()
plot.imshow(train_images[1])
plot.colorbar()
plot.grid(False)
plot.show()
"""

## BURASIIII

"""
plot.figure(figsize=(10,10))
for i in range(25):
    plot.subplot(5, 5, i+1)
    plot.xticks([])
    plot.yticks([])
    plot.grid(False)
    plot.imshow(train_images[i], cmap=plot.cm.binary)
    plot.xlabel(class_names[int(train_labels[i])]) # only integer scalar arrays is converted to a scalar array therefore int() is necessary
plot.show()
"""

"""SETUP THE LAYERS"""
#Flatten transforms the format of the images from 2d to 1d array.
#Parameter 128 of first dense is number of nodes(neurons)
#Parameter 10 of second dense is 10 probability, note that it equals to number of classes
#model = keras.Sequential([keras.layers.Flatten(input_shape = (32,32)), keras.layers.Dense(128, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])
y_train = keras.utils.to_categorical(train_labels, num_classes)
y_test = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


"""COMPILE THE MODEL"""
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


#BUU
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255


"""TRAIN THE MODEL"""
#Epochs is necessary for upgrading correct weights.
model.fit(train_images, train_labels, epochs = 5)
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy: ', accuracy)


"""MAKE PREDICTIONS"""

predictions = model.predict(test_images)
#print(class_names[np.argmax(predictions[0])]) ## returns a class name

i = 0
plot.figure(figsize=(6,3))
plot.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plot.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plot.show()
