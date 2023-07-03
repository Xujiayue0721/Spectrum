import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import dot
from sklearn.decomposition import PCA

from Preprocessing.Preprocessing import Preprocessing


def get_mahalanobis(x, i, j):
    xT = x.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.inv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A = x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return np.sqrt(dot(dot(tp, invD), tp.T))

df=pd.read_excel(r'D:\建模数据\固体粉末建模\建模数据.xlsx')
# yy=df.columns[197:198].tolist()
# print(yy)
x1=df.values[:,2:-1]
data_y=df.values[:,1:2]
pca = PCA(n_components=2)
pca = pca.fit(x1)
x_dr = pca.transform(x1)
print(x_dr)
plt.scatter(x_dr[:, 0], x_dr[:, 1], marker='o')
plt.show()
print("-----------")
arr=[]
for i in range (len(x_dr)):
    arr.append(get_mahalanobis(x_dr, i, -1))
print(arr)
plt.plot(arr, 'o')
plt.show()
