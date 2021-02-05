#AW=P და პიკაროს გასაშუალოებულების შედარება
#AW+B=P და პიკაროს გასაშუალოებულების შედარება
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

#print(CO_B2, CO_B4, CO_Pikaro)

i=0
left = len(CO_B2)
w11 = []
w12 = []
w21=[]
w22=[]
b=[]
w1_pair = []
w2_pair= []
M=range(556140)
diff_A1_P = []
diff_A2_P=[]
P_av=[]
while(i<len(CO_B2) and left>3600):
    CO_B2_temp = CO_B2[i:i+3600]
    CO_B4_temp = CO_B4[i:i+3600]
    CO_Pikaro_temp = CO_Pikaro[i:i+3600]
    CO_Boards_average_temp=CO_Boards_average[i:i+3600]
    A1 = np.c_[CO_B2[i:i+3600],CO_B4[i:i+3600]]
    A1_plus = np.linalg.pinv(A1)
    w1 = A1_plus.dot(CO_Pikaro_temp)
    w11.append(w1[0])
    w12.append(w1[1])
    w1_pair.append([w1[0],w1[1]])
    A1_average=sum(A1.dot(w1))/3600
    P_average=sum(CO_Pikaro_temp)/3600
    P_av.append(P_average)
    difference_A1_P_average=A1_average-P_average
    diff_A1_P.append(difference_A1_P_average)
    A2 = np.c_[CO_B2[i:i+3600],CO_B4[i:i+3600],np.ones(3600)]
    A2_plus = np.linalg.pinv(A2)
    w2 = A2_plus.dot(CO_Pikaro_temp)
    w21.append(w2[0])
    w22.append(w2[1])
    b.append(w2[2])
    w2_pair.append([w2[0],w2[1],w2[2]])
    A2_average=sum(A2.dot(w2))/3600
    difference_A2_P_average=A2_average-P_average
    diff_A2_P.append(difference_A2_P_average)
    i+=3600
    left = len(CO_B2) - i

P_av=np.array(P_av)
diff_A1_P=np.array(diff_A1_P)
diff_A2_P=np.array(diff_A2_P)


errorinf1=(abs(diff_A1_P)).max()/(abs(P_av)).max()
error1st1=np.linalg.norm(diff_A1_P, ord=1)/np.linalg.norm(P_av, ord=1)
error2nd1=np.linalg.norm(diff_A1_P, ord=2)/np.linalg.norm(P_av, ord=2)

errorinf2=(abs(diff_A2_P)).max()/(abs(P_av)).max()
error1st2=np.linalg.norm(diff_A2_P, ord=1)/np.linalg.norm(P_av, ord=1)
error2nd2=np.linalg.norm(diff_A2_P, ord=2)/np.linalg.norm(P_av, ord=2)

print("AW=P")
print(errorinf1)
print(error1st1)
print(error2nd1)
print("AW+B=P")
print(errorinf2)
print(error1st2)
print(error2nd2)

#(ჯერ თითოეული 60 წამისთვის განვიხილეთ პიკაროს საშუალოს(ჯამი/60) და პენროუზის საშუალოს სხვაობები, და შემდეგ ამ სხვაობებით შექმნილი ვექტორის ნორმები დავთვალეთ)