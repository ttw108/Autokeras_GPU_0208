import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdcm
import keras
import  keras.utils as np_utils
from sklearn.model_selection import train_test_split

xyd=pd.read_csv('XYdata-197x31_s.csv')
xyd.shape
#將Stress polarMap 轉成array
xydata=np.asarray(xyd)[:,1:]
xydata.shape #6107個*16385 x+y

x_6107 =xydata[:,:16384]
x_6107.shape
y_6107=xydata[:,16384:16385]
y_6107.shape



#轉成6107*128*128
x_128_2=x_6107.reshape(-1,128,128)
x_128_2.shape #197*128*128

#轉成6107*128*128*1
x_128_4D=x_128_2.reshape(-1,128,128,1)
x_128_4D.shape #6107*128*128*1
#################以上6107X  處理完畢


##############接下來換y########################################
y_4d=y_6107


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

import autokeras
from autokeras import ImageClassifier
if __name__ == '__main__':
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train.reshape(x_train.shape + (1,))
    #x_test = x_test.reshape(x_test.shape + (1,))
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=120 * 600 * 600)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)