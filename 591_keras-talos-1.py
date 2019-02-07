import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos import live
from keras.models import Sequential
from keras.layers import Dropout, Dense

# Keras items
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu
from keras.losses import binary_crossentropy
import matplotlib



import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdcm
import keras
import  keras.utils as np_utils
import sklearn
from sklearn.model_selection import train_test_split

xyd=pd.read_csv('XYdata-591-197x31_s.csv')
xyd.shape
#將Stress polarMap 轉成array
xydata=np.asarray(xyd)[:,1:]
xydata.shape #6107個*16385 x+y

x_591 =xydata[:,:16384]/255
x_591.shape
y_591=xydata[:,16384:16385]
y_591.shape

#轉成6107*128*128
x_128_2=x_591.reshape(-1,128,128)
x_128_2.shape #197*128*128

#轉成6107*128*128*1
x_128_4D=x_128_2.reshape(-1,128,128,1)
x_128_4D.shape #6107*128*128*1
#################以上6107X  處理完畢
##############接下來換y########################################
y_4d=y_591

# 设置数据集
x, y = x_128_4D, y_4d
x.shape
y=np_utils.to_categorical(y_4d.astype('int'))
y.shape

'''''
以上完成資料整理
接下來執行CNN建模
'''
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
# first we have to make sure to input data and params into the function
def polar_model(x_train, y_train, x_test, y_test, params):
    model = Sequential()
    # CNN1
    model.add(
        Conv2D(filters=24,kernel_size=(5, 5),
            padding='same',input_shape=(128, 128, 1),
            activation=params['activation']))
    # MaxPooling1
    model.add(MaxPool2D(pool_size=(2, 2)))
    # CNN2
    model.add(Conv2D(filters=32,kernel_size=(3, 3),
            padding='same',activation=params['activation']))
    # MaxPooling2
    model.add(MaxPool2D(pool_size=(2, 2)))
    # CNN3
    model.add(Conv2D(filters=64,kernel_size=(3, 3),
            padding='same',activation=params['activation']))
    # MaxPooling2
    model.add(MaxPool2D(pool_size=(2, 2)))
    # CNN4
    model.add(Conv2D(filters=64,kernel_size=(3, 3),
            padding='same',activation='relu'))
    # MaxPooling2
    model.add(MaxPool2D(pool_size=(2, 2)))
    # dropout 避免Overfitting
    model.add(Dropout(params['dropout']))
    # 平坦化
    model.add(Flatten())
    model.add(Dense(64, activation=params['activation']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(2, activation='softmax'))
    # 模型訓練
    lrate = 0.0005
    decay = lrate / params['epochs']
    adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                 epsilon=None, decay=decay, amsgrad=True)
    model.compile(loss=params['losses'],optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(
        x=x_train,y=y_train, validation_data=[x_test, y_test],
        callbacks=[live()],epochs=params['epochs'],batch_size=params['batch_size'],
        verbose=2)
    return history, model
# then we can go ahead and set the parameter space
p = {'first_neuron':[9,10,11],
     'hidden_layers':[0, 1, 2],
     'batch_size': [30,50],
     'epochs': [75,100],
     'dropout': [0.1,0.2,0.4],
     'kernel_initializer': ['uniform','normal'],
     'optimizer': [Nadam, Adam],
     'losses': [binary_crossentropy],
     'activation':[relu, elu],
     }

# and run the experiment
t = ta.Scan(x=x,
            y=y,
            model=polar_model,
            params=p,
            dataset_name='Polar_model',
            experiment_no='1')