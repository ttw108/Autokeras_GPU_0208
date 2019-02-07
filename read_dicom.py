import pydicom as pydcm
import matplotlib.pyplot as plt
import dicom as dcm
ds=pydcm.dcmread('./MPI all/R/0009_R.dcm')
#a=pdm.read_file('./MPI all/R/0009_R.dcm')
ds.dir

pt=ds.PatientName
ds.pixel_array.shape

##原始二进制文件
ps = ds.PixelData

##CT值组成了一个矩阵
pix = ds.pixel_array

##读取显示图片
pylab.imshow(ds.pixel_array,
             cmap=pylab.cm.bone)
pylab.show()

plt.imshow(ps)