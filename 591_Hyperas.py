from __future__ import print_function
import numpy as np

from hyperopt import STATUS_OK, tpe
from hyperopt import Trials
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

import os
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def data():
    xyd = pd.read_csv('XYdata-591-197x31_s.csv')
    xydata = np.asarray(xyd)[:, 1:]
    x_591 = xydata[:, :16384] / 255
    y_591 = xydata[:, 16384:16385]
    x_128_2 = x_591.reshape(-1, 128, 128)
    x_128_4D = x_128_2.reshape(-1, 128, 128, 1)

    y_4d = y_591
    x, y = x_128_4D, y_4d
    x.shape
    y.shape
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    y_train = np_utils.to_categorical(y_train.astype('int'))
    y_test = np_utils.to_categorical(y_test.astype('int'))
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, input_shape=(16384,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([16,32,64,128,256,512])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))
        # We can also choose between complete sets of layers
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([30,48,64,128])}},
              epochs=8,
              verbose=2,
              validation_split=0.3)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials = Trials)

    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)