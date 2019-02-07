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
y.shape
# x = array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
# y = range(0, 5)
# 切分数据，不固定随机种子（random_state）时，同样的代码，得到的训练集数据不同。

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)

x_train.shape
x_test.shape
y_train.shape
y_test.shape


y_trainOneHot=np_utils.to_categorical(y_train.astype('int'))
y_testOneHot=np_utils.to_categorical(y_test.astype('int'))

'''''
以上完成資料整理
接下來執行CNN建模
'''

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

model=Sequential()

#CNN1
model.add(
    Conv2D(
        filters=24,
        kernel_size=(5,5),
        padding='same',
        input_shape=(128,128,1),
        activation='relu'
        ))

#MaxPooling1
model.add(
    MaxPool2D(pool_size=(2,2)))

#CNN2
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        padding='same',
        activation='relu')
)

#MaxPooling2
model.add(
    MaxPool2D(
        pool_size=(2,2)))



#CNN3
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='same',
        activation='relu')
)

#MaxPooling2
model.add(
    MaxPool2D(
        pool_size=(2,2)))

#dropout 避免Overfitting
model.add(Dropout(0.4))
#平坦化
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

#模型訓練
lrate=0.0005
epochs=250
decay=lrate/epochs
sgd= keras.optimizers.SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
adam=keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              #optimizer=sgd,
              metrics=['accuracy'])

train_history=model.fit(
    x=x_train,
    y=y_trainOneHot,
    validation_split=0.25,
    epochs=epochs,
    batch_size=30,
    verbose=2
)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

show_train_history(train_history,'acc','val_acc')

show_train_history(train_history, 'loss','val_loss')

scores = model.evaluate(x_test, y_testOneHot)
scores[1]

import pandas as pd
prediction = model.predict_classes(x_test)

print(y_testOneHot.shape)
pd.crosstab(y_train, prediction, rownames=['label'], colnames=['predict'])

model.summary()





