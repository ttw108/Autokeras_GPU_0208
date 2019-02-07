#參考CNN-2 調整filter 3*3 pool 2*2
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

path = "./MPI-197/S/"
for filenames in os.walk(path):
    for filename in filenames:
        print(filename)
fn=filenames[2][::].copy()

#從fn用 for 迴圈讀檔名
alld = pd.DataFrame([])
for i in fn[0:]:
    print(i)
    #讀檔並取出image array
    ds = pdcm.dcmread(path+i)
    data = ds.pixel_array
    #轉為1維陣列
    d_1 = np.reshape(data, (1, np.product(data.shape)))
    #將d_1 norm 成0~1
    d1_max=np.max(d_1)
    d_1=d_1/d1_max
    df_1d=pd.DataFrame(d_1)
    #寫成資料DATAFRAME
    if fn.index(i) == 0:
        alldata = pd.concat([alld, df_1d], axis=0)
    else:
        alldata = pd.concat([alldata, df_1d], axis = 0)

alldata.shape

#將Stress polarMap 轉成array
ad=np.asarray(alldata)
ad.shape #197*16384

#轉成197*128*128
nad=ad.reshape(-1,128,128)
nad.shape #197*128*128

#轉成197*128*128*1
x_197=nad.reshape(197,128,128,1)
x_197.shape #197*128*128*1
b=nad[0] #確認形狀

y_197=pd.read_csv('MPI-197/y_197.csv')

# 设置数据集
x, y = x_197, y_197
x.shape
y.shape
# x = array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
# y = range(0, 5)
# 切分数据，不固定随机种子（random_state）时，同样的代码，得到的训练集数据不同。




############################################################################
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.1,
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
        filters=16,
        kernel_size=(10,10),
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
        filters=24,
        kernel_size=(5,5),
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
        filters=48,
        kernel_size=(5,5),
        padding='same',
        activation='relu')
)

#MaxPooling2
model.add(
    MaxPool2D(
        pool_size=(2,2)))

#dropout 避免Overfitting
model.add(Dropout(0.2))
#平坦化
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

#模型訓練
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history=model.fit(
    x=x_train,
    y=y_trainOneHot,
    validation_split=0.2,
    epochs=80,
    #batch_size=30,
    verbose=1
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





