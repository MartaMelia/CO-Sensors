import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import coolwarm
from sklearn.cluster import KMeans
from sklearn import datasets

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
m=0
n=0
k=0
l=0
v=0
q=0
t=0
r=0
p=0
f=0
w11 = []
w12 = []
b=[]
S_CO_Pikaro=[]
w1_pair = []
w2_pair = []
MAE=[]
w21=[]
w22=[]
Boards_predicted=[]
error1st=[]
error2nd=[]
errorinf=[]
error1st_new=[]
error2nd_new=[]
errorinf_new=[]
M=range(556140)
diff_A1_P = []
diff_A2_P = []
while(i<len(CO_B2) and left>16):
    CO_B2_temp = CO_B2[i:i+16]
    CO_B4_temp = CO_B4[i:i+16]
    CO_Pikaro_temp = CO_Pikaro[i:i+16]
    s=sum(CO_Pikaro[i:i+16])/16
    S_CO_Pikaro.append(s)
    CO_Boards_average_temp=CO_Boards_average[i:i+16]
    A1 = np.c_[CO_B2[i:i+16],CO_B4[i:i+16]]
    A2 = np.c_[CO_B2[i:i+16],CO_B4[i:i+16],np.ones(16)]
    A1_plus = np.linalg.pinv(A1)
    A2_plus = np.linalg.pinv(A2)
    w1 = A1_plus.dot(CO_Pikaro_temp)
    w11.append(w1[0])
    w12.append(w1[1])
    w2 = A2_plus.dot(CO_Pikaro_temp)
    w21.append(w2[0])
    w22.append(w2[0])
    b.append(w2[2])
    w1_pair.append([w1[0],w1[1]])
    w2_pair.append([w2[0],w2[1],w2[2]])
    #AW=P-ს ამონახსნის ცდომილების რამდენი პროცენტისთვის რამდენი წერტილი გვაქვს
    difference_A1_P=np.subtract(A1.dot(w1),CO_Pikaro_temp)
    diff_A1_P.extend(difference_A1_P)
    error1=np.linalg.norm(difference_A1_P, ord=1)/np.linalg.norm(CO_Pikaro_temp, ord=1)
    error1st.append(error1)
    error2=np.linalg.norm(difference_A1_P, ord=2)/np.linalg.norm(CO_Pikaro_temp, ord=2)
    error2nd.append(error2)
    erroroo=(abs(difference_A1_P)).max()/(abs(CO_Pikaro_temp)).max()
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
    #AW+B=P-ს ამონახსნის ცდომილების რამდენი პროცენტისთვის რამდენი წერტილი გვაქვს
    difference_A2_P=np.subtract(A2.dot(w2),CO_Pikaro_temp)
    diff_A2_P.extend(difference_A2_P)
    error1_new=np.linalg.norm(difference_A2_P, ord=1)/np.linalg.norm(CO_Pikaro_temp, ord=1)
    error1st_new.append(error1_new)
    error2_new=np.linalg.norm(difference_A2_P, ord=2)/np.linalg.norm(CO_Pikaro_temp, ord=2)
    error2nd_new.append(error2_new)
    erroroo_new=(abs(difference_A2_P)).max()/(abs(CO_Pikaro_temp)).max()
    if erroroo_new>0.4:
        q+=1
    elif erroroo_new>0.2:
        t+=1
    elif erroroo_new>0.1:
        r+=1
    elif erroroo_new>0.05:
        p+=1
    else : 
        f+=1
    errorinf_new.append(erroroo_new)
    mae=sum(difference_A1_P)/len(CO_B2)
    MAE.append(mae)
    i+=16
    left = len(CO_B2) - i

print('AW=P')
print(m)
print(n)
print(k)
print(l)
print(v)
print('AW+B=P')
print(q)
print(t)
print(r)
print(p)
print(f)


r'''
# წრფივი რეგრესია AW+B=P-ს მურ-პენროუზის ამონახსნისთვის w2=k1w1+k2b+c
X=pd.DataFrame({"w21":w21,"b":b})
Y=pd.DataFrame({"w22": w22})
regr = linear_model.LinearRegression()
regr.fit(X,Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
'''
r'''
# 3D ნახაზი მურ-პენროუზით მიღებული w1,w2,b ცვლადებისთვის
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
naxaz=ax.scatter3D(w21, w22, b, c=errorinf_new, cmap='hot', marker='o')
fig.colorbar(naxaz, ax=ax)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('b')
'''

# 3D ნახაზი წრფივი რეგრესიისთვის w1.w2,b
w21=np.array(w21)
b=np.array(w22)
def myfunc(w21,b):
  return w21*(1.00000000e+00)+b*(-2.36650813e-16)+8.8817842e-15
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nax=ax.scatter3D(w21, myfunc(w21,b), b, c=errorinf_new, cmap='hot', marker='o')
fig.colorbar(nax, ax=ax) 
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('b')


r'''
# AW=P-ს და AW+B=P-ს ამონახსნებს შორის ცდომილება
diff_A=np.subtract(A1.dot(w1),A2.dot(w2))
AbsError1st=np.linalg.norm(diff_A, ord=1)/np.linalg.norm(A1.dot(w1), ord=1)
AbsError2nd=np.linalg.norm(diff_A, ord=2)/np.linalg.norm(A1.dot(w1), ord=2)
AbsErroroo=(np.abs(diff_A)).max()/(np.abs(A1.dot(w1))).max()
print(AbsError1st)
print(AbsError2nd)
print(AbsErroroo)
'''
iris = datasets.load_iris()
X = iris.data
Y = iris.target
plt.figure('Iris dataset', figsize=(7,5))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2],c=Y)
k_means = KMeans(n_clusters=3)
k_means.fit(X)
k_means_predicted = k_means.predict(X)
accuracy = round((np.mean(k_means_predicted==Y))*100)
centroids = k_means.cluster_centers_
target_names = iris.target_names
colors = ['navy', 'turquoise', 'darkorange']
plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X[:,3],X[:,0],X[:,2], c=Y , cmap='Set2', s=50)

# color missclassified data

ax.scatter(X[k_means_predicted!=Y,3],X[k_means_predicted!=Y,0],X[k_means_predicted!=Y,2] ,c='b', s=50)

# plot centroids

ax.scatter(centroids[0,3],centroids[0,0],centroids[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centroids[1,3],centroids[1,0],centroids[1,2] ,c='r', s=50)
ax.scatter(centroids[2,3],centroids[2,0],centroids[2,2] ,c='r', s=50)

ax.legend()