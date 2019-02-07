import numpy as np
import sklearn
from keras.datasets import mnist
import keras
# from autokeras.image_supervised import ImageClassifier
from autokeras import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train.shape
    x_train = x_train.reshape(x_train.shape + (1,))
    x_train.shape
    x_test = x_test.reshape(x_test.shape + (1,))

    clf = ImageClassifier(verbose=True)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.fit(x_train, y_train)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y)
