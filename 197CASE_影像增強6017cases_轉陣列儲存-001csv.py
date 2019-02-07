import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdcm
import keras
import  keras.utils as np_utils
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.misc import toimage
import image as img
import imageio
path = "./MPI-197/S/"
for filenames in os.walk(path):
    for filename in filenames:
        print(filename)
fn=filenames[2][::].copy()

##############讀取-y_197.csv#########################
y_197=pd.read_csv('MPI-197/y_197.csv')

y_197=np.asarray(y_197)
#從fn用 for 迴圈讀檔名
alld = pd.DataFrame([])
allpt=pd.DataFrame([])
for i in fn[0:]:
#for i in fn[0:]:
    print(i)
    #讀檔並取出image array
    ds = pdcm.dcmread(path+i)
    data = ds.pixel_array
    data_c=data/np.max(data)*255
    data_int=data_c.astype(np.int32)
#########################################以上轉成0-255的array

############  準備轉成image  #######################
    arr = data_int
    #arr = arr.reshape(height, width)
    im = Image.fromarray(arr)
    im = toimage(arr)
    #plt.imshow(im)

#########轉角度##################################################
    allr= pd.DataFrame([])
    all_ro_db=pd.DataFrame([])
    all_db=pd.DataFrame([])
    for j in range(0,31,15):
        im_r=im.rotate(j, Image.BILINEAR )
        #plt.imshow(im45)
        imc = np.asarray(im_r,dtype='int');
        rd_1 = np.reshape(imc, (1, np.product(imc.shape)))
        rd_11=np.append(rd_1,y_197[fn.index(i)])
        df_1d=pd.DataFrame(rd_11)

        if j == 0:
            all_ro_db = pd.concat([allr, df_1d], axis= 1)
        else:
             all_ro_db  = pd.concat([all_ro_db , df_1d], axis = 1)

    all_ro_db.shape
    allpt=pd.concat([allpt,all_ro_db], axis = 1)
    allpt.shape
       # locals()["all_ro_db"+str(i)]=pd.DataFrame([])
        #locals()["all_ro_db"+str(i)]=pd.concat([locals()["all_ro_db"+str(i)], df_1d], axis= 1)




allpt.shape
alldata=np.asarray(allpt)
alldata.shape
alldataT=alldata.T
alldataT.shape
alldataT=pd.DataFrame(alldataT)
alldataT.shape

pd.DataFrame.to_csv (alldataT,'XYdata-591-197x31_s.csv')




















