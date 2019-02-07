import numpy as np
import pandas as pd

a=[[1,2,3],
   [4,5,6]]

b=[[7,8,9],
   [10,11,12]]

c=np.append(a,b)

d=c.reshape(-1,2,2)

np.asarray(a)
np.shape(a)



np.str(a)
np.shape(a)
np.shape(b)
c=a+b
np.shape(c)

a.append(b)

myarray=np.asarray(a)
np.shape(a)


a=list(range(1,20))
max(a)