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
m=0
n=0
k=0
l=0
q=0
t=0
r=0
p=0
v=0
w1 = []
w2 = []

w_pair = []
error1st=[]
error2nd=[]
errorinf=[]
MAE=[]
w2_new=[]
Boards_predicted=[]
w_pair_new=[]
M=range(556140)
diff_A_P = []
while(i<len(CO_B2) and left>60):
    CO_B2_temp = CO_B2[i:i+60]
    CO_B4_temp = CO_B4[i:i+60]
    CO_Pikaro_temp = CO_Pikaro[i:i+60]
    CO_Boards_average_temp=CO_Boards_average[i:i+60]
    A = np.c_[CO_B2[i:i+60],CO_B4[i:i+60]]
    A_plus = np.linalg.pinv(A)
    w = A_plus.dot(CO_Pikaro_temp)
    w1.append(w[0])
    w2.append(w[1])
    w_pair.append([w[0],w[1]])
    difference_A_P=np.subtract(A.dot(w),CO_Pikaro_temp)
    difference_CO_B_P=np.subtract(CO_Boards_average_temp,CO_Pikaro_temp)
    eoo=abs(difference_CO_B_P).max()/abs(CO_Pikaro_temp).max()
    if eoo>0.4:
        q+=1
    elif eoo>0.2:
        t+=1
    elif eoo>0.1:
        r+=1
    else : 
        p+=1
    #sum(CO_Pikaro_temp)/60
    diff_A_P.extend(difference_A_P)
    error1=np.linalg.norm(difference_A_P, ord=1)/np.linalg.norm(CO_Pikaro_temp, ord=1)
    error1st.append(error1)
    error2=np.linalg.norm(difference_A_P, ord=2)/np.linalg.norm(CO_Pikaro_temp, ord=2)
    error2nd.append(error2)
    erroroo=(abs(difference_A_P)).max()/(abs(CO_Pikaro_temp)).max()
    if erroroo>0.4:
        m+=1
    elif erroroo>0.2:
        n+=1
    elif erroroo>0.1:
        k+=1
    elif erroroo>0.05:
        l+=1
    else : 
        v+=1
    errorinf.append(erroroo)
    #print(error1,' ', error2,' ',erroroo)
    mae=sum(difference_A_P)/len(CO_B2)
    MAE.append(mae)
    
    i+=60
    left = len(CO_B2) - i
    

w_pair=np.array(w_pair)   


print(m)
print(n)
print(k)
print(l)
print(v)
r'''
print(q)
print(t)
print(r)
print(p)
'''
r'''
plt.plot(MAE,'b.')
plt.show()
plt.plot(error1st,'b.')
plt.show()
plt.plot(error2nd,'b.')
plt.show()
plt.plot(errorinf,'b.')
plt.show()
'''

w=[w1,w2]
slope, intercept, r, p, std_err = stats.linregress(w1, w2)
#print(slope,intercept,r,p,std_err)
'''
def myfunc(w1):
  return slope * w1 + intercept
mymodel = list(map(myfunc, w1))
#coefficient of correlation
#print(r)
#k and b
#print(slope)
#print(intercept)
plt.plot(w1,mymodel,'r')
#plt.ylabel('w2')
#plt.xlabel('w1')
#plt.scatter(w1,w2)
#plt.show()
plt.plot(w1,w2,'b.')
plt.ylabel('w2')
plt.xlabel('w1')
plt.show()


kmeans = KMeans(n_clusters=10)
kmeans.fit(w_pair)
plt.scatter(w_pair[:,0],w_pair[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()

#print(w_pair[:,0])

#centroid coordinates
(kmeans.cluster_centers_)


unit_w1 = w1 / np. linalg. norm(w1)
unit_w2= w2/ np. linalg. norm(w2)
dot_product = np. dot(unit_w1, unit_w2)
angle = np. arccos(dot_product)
cosa=math.cos(angle)
#print(cosa)

#plt.scatter(M,diff_A_P,)
#plt.show()

#print(diff_A_P)
#print(MAE)
#plt.plot(diff_A_P,'b')
#plt.show()
#plt.plot(MAE,'r.')
#plt.show()
'''

r'''
#მეორე ნაწილი
A_new=[]
i=0
left=len(CO_B2)
while(i<len(CO_B2) and left>60):
    A_new = np.c_[CO_B2[i:60+i],CO_B4[i:60+i]]
    w2_new.append(slope*w1[int(i/60)]+intercept)
    w_pair_new.append([w1[int(i/60)],w2_new[int(i/60)]])
    Boards_predicted.append(A_new.dot(w_pair_new[int(i/60)]))
#    print(len(Boards_predicted[0]))
    i+=60
    left = len(CO_B2) - i
Boards_predicted = np.array(Boards_predicted)
Boards_predicted = Boards_predicted.flatten()
CO_Pikaro_trimmed = CO_Pikaro[0:556140]
diff_Pikaro_Boardspredicted=np.subtract(Boards_predicted,CO_Pikaro_trimmed)

AbsError1st=np.linalg.norm(diff_Pikaro_Boardspredicted, ord=1)/np.linalg.norm(CO_Pikaro_trimmed, ord=1)/556140
AbsError2nd=np.linalg.norm(diff_Pikaro_Boardspredicted, ord=2)/np.linalg.norm(CO_Pikaro_trimmed, ord=2)
AbsErroroo=(np.abs(diff_Pikaro_Boardspredicted)).max()/(np.abs(CO_Pikaro_trimmed)).max()
print(AbsError1st)
print(AbsError2nd)
print(AbsErroroo)
#print(w2_new)  
#plt.plot(w1,w2_new)

plt.plot(Boards_predicted)
plt.show()
plt.plot(CO_Pikaro_trimmed)
plt.show()
'''

               