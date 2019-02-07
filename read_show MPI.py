import os
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdcm

# path = "./MPI all/S/"
# for filenames in os.walk(path):
#     for filename in filenames:
#         print(os.path.join(dirpath, filename))
# print(filename)
# filename[]


path = "./MPI all/S/"
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
    df_1d=pd.DataFrame(d_1)
    #寫成資料DATAFRAME
    if fn.index(i) == 0:
        alldata = pd.concat([alld, df_1d], axis=0)
    else:
        alldata = pd.concat([alldata, df_1d], axis = 0)
a_index=list(range(1,145))
print(a_index)
inm=str(a_index)

alldata.index.name = "Pt No."
alldata.index=list(range(1,146))
alldata.columns=list(range(1,16385))
alldata.to_csv('xdata-145_s.csv')

alldata.shape

    # plt.imshow(data)
    #     # plt.colorbar()
    #     # plt.show()






# #####################################################
#參考用~~~~

# #讀dicom檔
# ds= pdcm.dcmread('./MPI all/R/0009_R.dcm')
# #讀dicom的array data
# data=ds.pixel_array
# #看陣列維度
#
# data.shape
#
# d_1= np.reshape(data, (1,np.product(data.shape)))
# plt.imshow(data)
# plt.colorbar()
# plt.show()
#
