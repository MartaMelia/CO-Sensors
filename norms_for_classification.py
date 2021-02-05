import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

location = r"C:\Users\MARTA\Documents\Alphasense\02.10-02.19.mnt.csv"

df = pd.read_csv(location)
CO_B2 = df.iloc[:,7]
CO_B4 = df.iloc[:,8]
CO_Pikaro = df.iloc[:,9]
CO_B2 = CO_B2.to_numpy()
CO_B4 = CO_B4.to_numpy()
BA=np.add(CO_B2,CO_B4)/2
CO_Pikaro = CO_Pikaro.to_numpy()
diff_BA_P= np.subtract(CO_Pikaro,CO_B2)
AbsError1st=np.linalg.norm(diff_BA_P, ord=1)/np.linalg.norm(CO_Pikaro, ord=1)
AbsError2nd=np.linalg.norm(diff_BA_P, ord=2)/np.linalg.norm(CO_Pikaro, ord=2)
AbsErroroo=erroroo=(abs(diff_BA_P)).max()/(abs(CO_Pikaro)).max()
MAE=sum(diff_BA_P)/len(CO_B2)
print(AbsError1st)
print(AbsError2nd)
print(AbsErroroo)
print(MAE)
