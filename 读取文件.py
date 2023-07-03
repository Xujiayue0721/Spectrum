# # import os
# #
# # import numpy as np
# # import pandas as pd
# # from matplotlib import pyplot as plt
# # def sum():
# #     file_directory = r'C:\Users\奎奎\Desktop\2021-12-13 乙醇水溶液dlp'  # 存放要合并的数据源文件路径
# #     # 存放每个excel数据
# #     excel_datas = []
# #     for root, dirs, files in os.walk(file_directory):  # 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
# #         for file in files:
# #             file_path = os.path.join(root, file)
# #             re_ex = pd.read_csv(file_path)  # 将excel转换成DataFrame
# #             # re_ex.drop(['0'])
# #             # print(re_ex)
# #             x1=np.sum(re_ex,axis=1)
# #             excel_datas.append(x1)
# #     all_datas = pd.concat(excel_datas) # 将所有DataFrame合成一个
# #     all_datas.to_csv(r'C:\Users\奎奎\Desktop\2021-12-13 乙醇水溶液dlp\sum.csv',index=False)
# #     data=pd.read_csv(r'C:\Users\奎奎\Desktop\2021-12-13 乙醇水溶液dlp\sum.csv')
# #
# # if __name__=="__main__":
# #     sum()
# #
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from numpy import genfromtxt
# # mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名
# # data=genfromtxt(r'D:\建模数据\参比_sum.csv',delimiter=',')
# # print(data)
# # c_data=data[1:,-1]
# # print(c_data)
# # m_list=[]
# # m_arr=np.array(m_list)
# # i=0
# # for info in os.listdir(r'D:\建模数据\2022-03-15 葡萄糖溶液建模'):
# #     domain = os.path.abspath(r'D:\建模数据\2022-03-15 葡萄糖溶液建模') #获取文件夹的路径
# #     info = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
# #     print(info)
# #     data = pd.read_csv(info,usecols=[1],header=40)
# #     new_data = data.iloc[:216, -1]
# #     # new_data=c_data-new_data#计算吸收光谱
# #     # print(new_data)
# #     # 把所有的数据放到一个csv文件当中
# #     df = pd.read_csv(r'D:\建模数据\透射_sum.csv')
# #     i+=1
# #     a='Intensity (AU)'+str(i)
# #     df[a] = new_data
# #     df.to_csv(r'D:\建模数据\透射_sum.csv', index=None)
# # cy_data=[0.001000315,0.001997235,0.002990934,0.003996732,0.004992414,0.006000321,0.006988847,0.008007359,
# # 0.008998942,0.010011892,0.011004569,0.012007146,0.013011634,0.013990707,0.015002287,0.016003,
# # 0.017005795,0.018005401,0.019001726,0.019999384,0.019507426,0.01853079,0.017509328,0.016442587,
# # 0.015529941,0.014514389,0.013460125,0.012476472,0.011488077,0.010493156,0.00947703,0.008490286,
# # 0.007494703,0.006370615,0.005495908,0.004513594,0.003497608,0.002498006,0.001498634,0.000500921,]
# # cy_data.sort()
# # print(cy_data)
# # y_data=[]
# # for c in cy_data:
# #     for j in range(10):
# #         y_data.append(c)
# def autoscaling(x):  # SNV标准正太变换
#     _std_ = np.std(x, ddof=1)  # ### std c = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))  axis =1 是针对行求方差
#     # ################
#     _x_bar_ = np.mean(x)  # 求一列元素的平均值
#     _x_ = (x - _x_bar_) / _std_
#     return _x_
# def MaxMinNormalize(x) -> object:  # ############## 输入的数据必须是以列排列的，每一列是一组光谱数据   axis=0表示取每一列的最值
#     _Max_ = np.max(x, axis=0)
#     _Min_ = np.min(x, axis=0)
#     _x_ = (x - _Min_) / (_Max_ - _Min_)
#     return _x_
# data=genfromtxt(r'D:\建模数据\参比_sum.csv',delimiter=',')
# # print(data)
# x_data=data[1:,0]
# new_data=data[1:,-1]
# # new_data=autoscaling(new_data)
# # new_data=MaxMinNormalize(new_data)
# print(new_data)
# plt.plot(x_data,new_data)
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # plt.title("Normalized absorption spectrum",fontsize='20') #添加标题
# plt.xlabel("Wavelength(nm)")
# plt.ylabel('Reference')
# plt.show()
# plt.savefig(r'D:\建模数据\论文图表')
# #
# # print(new_data.mean(axis=1))
# # df = pd.read_csv(r'D:\建模数据\参比_sum.csv')
# # df['avg'] = new_data.mean(axis=1)
# # df.to_csv(r'D:\建模数据\参比_sum.csv', index=None)
# # data=genfromtxt(r'D:\建模数据\参比_sum.csv',delimiter=',')
import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_excel(r"C:\Users\奎奎\Desktop\预测结果.xlsx")
data1=df1.values[:,1]
data2=df1.values[:,2]
# data3=df1.values[:,2]
error=data2-data1
print(error)
error1=abs(error)
error1.sort()
print(error1)
# data4=df1.values[:,3]
# data5=df1.values[:,4]
# data6=df1.values[:,5]

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(data1,label="实际值")
plt.plot(data2,label="选取波长")
plt.legend()
plt.show()
plt.plot(data1,label="实际值")
plt.plot(data2,label="选取波长")
plt.legend()
plt.show()
plt.xlabel("浓度")
plt.ylabel('误差')
plt.scatter(data1,error)
plt.show()
# plt.plot(data1,label="实际值")
# plt.plot(data4,label="选取波长特征筛选")
# plt.legend()
# plt.show()
# plt.plot(data1,label="实际值")
# plt.plot(data5,label="全波长特征筛选")
# plt.legend()
# plt.show()
