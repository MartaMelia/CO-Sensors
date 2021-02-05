import pandas as pd
import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

location = r"C:\Users\MARTA\Documents\Alphasense\02.10-02.19.sec.csv.csv"

df = pd.read_csv(location)
CO_B2 = df.iloc[:,7]
CO_B4 = df.iloc[:,8]
CO_Pikaro = df.iloc[:,9]
CO_B2 = CO_B2.to_numpy()
CO_B4 = CO_B4.to_numpy()
CO_Pikaro = CO_Pikaro.to_numpy()
CO_Boards_average=np.add(CO_B2,CO_B4)/2


i=0
left = len(CO_B2)
w1 = []
w2 = []
w_pair = []
w2_new=[]
error1st=[]
error2nd=[]
errorinf=[]
MAE=[]
M=range(556140)
diff_A_P = []
while(i<len(CO_B2) and left>60):
    CO_B2_temp = CO_B2[i:i+60]
    CO_B4_temp = CO_B4[i:i+60]
    CO_Pikaro_temp = CO_Pikaro[i:i+60]
    CO_Boards_average_temp=CO_Boards_average[i:i+60]
    A = np.c_[CO_B2[0:60],CO_B4[0:60]]
    A_plus = np.linalg.pinv(A)
    w = A_plus.dot(CO_Pikaro_temp)
    w1.append(w[0])
    w2.append(w[1])
    w_pair.append([w[0],w[1]])
    i+=60
    left = len(CO_B2) - i
    

w_pair=np.array(w_pair)   
w=[w1,w2]
left = len(CO_B2)
i=0

slope, intercept, r, p, std_err = stats.linregress(w1, w2)
while(i<len(CO_B2) and left>60):
    #print(i)
    temp=int(i/60)
    w[0][temp]=w[0][temp]*slope+intercept
    w2_new.append(w[0][temp])
    i+=60
    left = len(CO_B2) - i

#მურ-პენროუზის ამონახსნსა და წრფივი რეგრესიით მიღებული შედეგს შორის ცდომილება
w_new=[w1,w2_new]
P1=A.dot(w_new)
P2=A.dot(w)
#print(w2_new)
#print(w_new)
diff_P=np.subtract(P1,P2)
AbsError1st=np.linalg.norm(diff_P, ord=1)/np.linalg.norm(P1, ord=1)
AbsError2nd=np.linalg.norm(diff_P, ord=2)/np.linalg.norm(P1, ord=2)
AbsErroroo=(np.abs(diff_P)).max()/(np.abs(P1)).max()
print(AbsError1st)
print(AbsError2nd)
print(AbsErroroo)