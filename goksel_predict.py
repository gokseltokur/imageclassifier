import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plot

from tensorflow import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import *
from keras.optimizers import *
import goksel_train as tr

test_data = 'testdata'
def testim():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        test_images.append([np.array(img)])
    return test_images


modelpath = 'model/gkslmodel.hdf5'
model = tr.createModel()
model.load_weights(modelpath)

testimages = tr.testdatalabel()
#testimages = testim()

test_images = np.array([i[0] for i in testimages]).reshape(-1,64,64,1)
test_labels = np.array([i[1] for i in testimages])


def predictimg():
    for count, data in enumerate(testimages):
        img = data[0]
        data = img.reshape(1, 64, 64, 1)
        modelout = model.predict([data])
        print(modelout)
        if np.argmax(modelout) == 1:
            cv2.imshow("Bicycle", img)
        else:
            cv2.imshow("Car", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    predictimg()
