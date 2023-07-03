from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from Preprocessing.Preprocessing import Preprocessing
import numpy as np
import pandas as pd
from numpy import genfromtxt, dot
from sklearn.model_selection import train_test_split
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Model.model import Model

df1 = pd.read_excel('./data/建模数据.xlsx')
df2 = pd.read_excel('./data/预测数据.xlsx')
yy = df1.columns[2:].tolist()
x1 = df1.values[:, 2:]
x2 = df2.values[:, 2:]
y1 = df1.values[:, 1:2]
y2 = df2.values[:, 1:2]
x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)
print(y1)

x_p = x.T
plt.plot(yy, x_p)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.title("原始光谱",fontsize='20') #添加标题
plt.xlabel("Wavelength(nm)")
plt.ylabel('杂质')
plt.show()
# 预处理
X = x.astype('float')
X = Preprocessing("SNV", X)
X = Preprocessing("SG", X)
# 特征筛选
FeatrueData, labels = SpctrumFeatureSelcet("Lars", X, y)
a = X[0, :]
b = FeatrueData[0, :]
idx = np.where(np.in1d(a, b))
idx = list(idx)
idx1 = np.array(idx)
yy = np.array(yy)
for i in range(len(idx)):
    print(yy[idx1[i]])
# idx=np.in1d(a, b,invert=True)
print(idx)
print(X[0, :])
print("特征筛选后")
print(FeatrueData.shape)
print(FeatrueData[0, :])
# 预处理后光谱图像
x_yy = X.T
plt.plot(yy, x_yy)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("{0}后".format('SNV'), fontsize='20')  # 添加标题
plt.xlabel("Wavelength(nm)")
# plt.ylabel('杂质')
plt.show()

arr_data = np.array([[0] * 1] * len(yy))


def get_mahalanobis(x, i, j):  # 马氏距离
    xT = x.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.inv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A = x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return (dot(dot(tp, invD), tp.T))


# 数据集划分
# x_train_vali, x_test, y_train_vali, y_test = train_test_split(FeatrueData, labels, test_size = 0.20, random_state = 10)
# x_train_vali, x_test, y_train_vali, y_test =SetSplit("random", FeatrueData, labels, test_size=0.2, randomseed=10)
x_train_vali = FeatrueData[:80, :]
x_test = FeatrueData[80:, :]

y_train_vali = labels[:80, :]
y_test = labels[80:, :]
pca1 = PCA(n_components=8)
pca1 = pca1.fit(x_train_vali)
x_dr = pca1.transform(x_train_vali)
arr = []
for i in range(len(x_dr)):
    arr.append(get_mahalanobis(x_dr, i, -1))
n = len(x_dr)
k = []  # 杠杆值数组
print("--------杠杆值-------")
for i in range(len(arr)):
    nn = (arr[i] / (n - 1)) + (1 / n)
    k.append(nn)
    if (nn > (3 * 8) / n):
        print(nn)

plt.plot(arr, 'o')
plt.title("马氏距离", fontsize='20')
plt.show()
plt.axhline(y=(3 * 8) / n, ls='--', c='blue')  # 添加水平线
plt.plot(k, 'o')
plt.title("杠杆值", fontsize='20')
plt.show()

# 进行预测
py_test, py_pred = Model("PLSR", x_train_vali, x_test, y_train_vali, y_test)  # PLSR模型


def getSlope(n, x, y):
    return (n * np.sum(x * y) - (np.sum(x) * np.sum(y))) / \
           (n * np.sum(x ** 2) - (np.sum(x)) ** 2)


def getIntercept(n, x, y):
    m = getSlope(n, x, y)
    return (np.sum(y) - m * np.sum(x)) / n


n = len(py_test)
y1 = getSlope(n, py_test, py_pred) * py_test + getIntercept(n, py_test, py_pred)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(py_test, py_pred, c='#7B68EE')
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.plot([0, 35], [0, 35], linestyle='dashed', c='black')
ax.plot(py_test, y1, c='#00F5FF')
plt.show()

# #各个模型对比图
# x = range(2,4,1)
# y = range(2,4,1)
# plt.plot(x,y)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.subplots_adjust(bottom=0.15)
# plt.subplots_adjust(left=0.15)
# plt.scatter(py_test,py_pred,label='PLSR_test',marker='2',color='g')
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.title("Comparison of prediction results",fontsize='20') #添加标题
# plt.xlabel("observed",fontsize=18)
# plt.ylabel('predicted',fontsize=18)
# plt.plot([0,35],[0,35],linestyle='dashed',c='black')
# plt.legend()
# plt.show()
