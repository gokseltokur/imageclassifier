import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plot

from tensorflow import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


train_data = 'traindata'
test_data = 'testdata'

def onehotlabel(img):
    label = img.split('.')[0]
    if label == 'car':
        ohl = np.array([1,0])
    elif label == 'bicycle':
        ohl = np.array([0,1])
    return ohl


def traindatalabel():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        train_images.append([np.array(img), onehotlabel(i)])
    shuffle(train_images)
    return train_images

def testdatalabel():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img), onehotlabel(i)])
    return test_images

def createModel():
    model = Sequential()
    model.add(InputLayer(input_shape=[64,64,1]))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu' ))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu' ))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu' ))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2, activation='softmax'))
    return model

def train():
    print(tf.__version__)

    trainimages = traindatalabel()
    testimages = testdatalabel()
    train_images = np.array([i[0] for i in trainimages]).reshape(-1,64,64,1)
    train_labels = np.array([i[1] for i in trainimages])
    test_images = np.array([i[0] for i in testimages]).reshape(-1,64,64,1)
    test_labels = np.array([i[1] for i in testimages])


    class_names = ['car', 'bicycle']
    num_classes = len(class_names)


    """SETUP THE LAYERS"""
    #Flatten transforms the format of the images from 2d to 1d array.
    #Parameter 128 of first dense is number of nodes(neurons)
    #Parameter 10 of second dense is 10 probability, note that it equals to number of classes
    #model = keras.Sequential([keras.layers.Flatten(input_shape = (32,32)), keras.layers.Dense(128, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])
    #y_train = keras.utils.to_categorical(train_labels, num_classes)
    #y_test = keras.utils.to_categorical(test_labels, num_classes)

    model = createModel()
    optimizer = Adam(lr=1e-3)

    """COMPILE THE MODEL"""
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])


    """TRAIN THE MODEL"""
    #Epochs is necessary for upgrading correct weights.
    model.fit(x=train_images, y=train_labels, epochs = 1000, batch_size=100)
    loss, accuracy = model.evaluate(test_images, test_labels)
    print('Test accuracy: ', accuracy)
    print('Test loss: ', loss)

    modelpath = 'model/gkslmodel.hdf5'
    model.save(modelpath)



if __name__ == '__main__':
    train()
