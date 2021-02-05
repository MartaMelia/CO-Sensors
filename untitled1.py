import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

location = r"C:\Users\MARTA\Documents\Alphasense\02.10-02.19.mnt.csv"

df = pd.read_csv(location)
CO_B2 = df["CO_B2_mnt_avg"]
CO_B4 = df["CO_B4_mnt_avg"]
CO_Boards_avg=np.add(CO_B2,CO_B4)/2
CO_Picarro = df["CO_Picaro_mnt_avg"]
CO_B2 = CO_B2.to_numpy()
CO_B4 = CO_B4.to_numpy()
CO_Picarro = CO_Picarro.to_numpy()
CO_B2_temp=CO_B2[0:1440]
CO_B4_temp =CO_B4[0:1440]
CO_Boards_avg_temp=CO_Boards_avg[0:1440]
CO_Picarro_temp=CO_Picarro[0:1440]
plt.plot(CO_B2_temp,'b.')
plt.ylabel("CO_B2")
plt.show()
plt.plot(CO_B4_temp,'b.')
plt.ylabel("CO_B4")
plt.show()
plt.plot(CO_Picarro_temp,'b.')
plt.ylabel("CO_Picarro")
plt.show()
plt.plot(CO_Boards_avg_temp,'b.')
plt.ylabel("CO_Boards_average")
plt.show()
