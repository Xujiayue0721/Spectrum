import joblib
import sklearn
from scipy import signal
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, PredefinedSplit

import numpy as np
import pandas as pd
from Preprocessing.Preprocessing import Preprocessing
from numpy import genfromtxt, dot
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

def get_mahalanobis(x, i, j):
    xT = x.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.inv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A = x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return (dot(dot(tp, invD), tp.T))

df1=pd.read_excel(r'D:\建模数据\杂质预测数据\建模数据-杂质(1).xlsx')#预测数据地址
yy=df1.columns[2:].tolist()
x=df1.values[197:,2:]
y=df1.values[197:,1:2]
x=x.astype('float')
# x=Preprocessing("SGDIFF",x)
x=Preprocessing("SNV",x)
x=np.array(x)
# x=Preprocessing("SG",x)
# x=Preprocessing("CT",x)
a=[14,  18,  20,  39,  45,  46,  56,  58,  60,  76,  87,  90, 102,
       104, 108, 112, 114, 117, 118, 128, 137, 141, 143, 150, 152, 157,
       159, 166, 169, 170, 177, 178, 182, 188, 193, 194, 198, 201, 202,
       207]
for i in a:
    print(yy[i])
x=x[:,a]
new_model = joblib.load("saved_model/pls_model6.pkl")#模型地址
# # 使用加载生成的模型预测新样本
pre_data=new_model.predict(x)
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(y, pre_data)))
d=np.average(y-pre_data)
arr=(y-pre_data-d)*(y-pre_data-d)
ans=0
for i in range(len(arr)-1):
    ans+=arr[i]
ans=float(abs(ans))
sse=np.sqrt(ans/(len(pre_data)-1))
print('SSE:', sse)
sqrtn=np.sqrt(len(y))
t=abs((abs(d)*sqrtn)/sse)
print("t:",t)
print('相关系数:',r2_score(y, pre_data))
print('RPD:',1/(np.sqrt(1-r2_score(y, pre_data))))
#对比图
def getSlope(n, x, y):
    return (n * np.sum(x * y) - (np.sum(x) * np.sum(y))) / \
           (n * np.sum(x ** 2) - (np.sum(x)) ** 2)


def getIntercept(n, x, y):
    m = getSlope(n, x, y)
    return (np.sum(y) - m * np.sum(x)) / n
n=len(y)
y1 = getSlope(n, y,pre_data) * y + getIntercept(n, y,pre_data)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(y,pre_data, c='#7B68EE')
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.plot([2.5,3.5],[2.5,3.5],linestyle='dashed',c='black')
ax.plot(y,y1, c='#00F5FF')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
#马氏距离
pca1=PCA(n_components=8)
pca1 = pca1.fit(x)
x_dr = pca1.transform(x)
arr=[]#马氏距离数组
for i in range (len(x_dr)):
    arr.append(get_mahalanobis(x_dr, i, -1))
ev=np.sum(pre_data-y)/(len(y))
print("ev:",ev)
ei_ev=(((pre_data-y)-ev)*((pre_data-y)-ev))
SDV=np.sqrt(np.sum(ei_ev)/(len(y)-1))
print("SDV:",SDV)
t=(abs(ev)*np.sqrt(len(y)))/SDV
print("t",t)
#求解SEC
secans=np.sum((pre_data-y)*(pre_data-y))
SEC=np.sqrt(secans/(len(y)-8))
print("SEC:",SEC)
print(pre_data)
#一致性验证
print("-----------")
lbarr=[]
uparr=[]
for n in range(len(arr)):
       lbarr.append(pre_data[n]-(t*SEC*np.sqrt(1+arr[n])))
       uparr.append(pre_data[n]+(t*SEC*np.sqrt(1+arr[n])))
for yy in range(len(y)):
       if(y[yy]<lbarr[yy] or y[yy]>uparr[yy]):
              print(y[yy])
plt.plot(lbarr)
plt.plot(uparr)
plt.plot(y, 'o')
plt.show()
