import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d


def Spline( x, y, x_min, x_max, n):  # 三次样条插值
    _x_ = np.array(x)
    _y_ = np.array(y)  # 取出y中的一列
    x_smooth = np.linspace(x_min, x_max, n)  # np.linspace 等差数列,从x.min()到x.max()生成750个数，便于后续插值
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth  # 返回插值后的光谱数据

def handler_multiple_lists(lists):
    """
    提出多维矩阵中重复的元素
    param lists : 多个列表组成的矩阵, [[1, 2, 3], [2, 3, 4], [4, 5, 6]]
    return : 重复数据组成的字段，键为重复的数据，值为出现的次数 例如 {"1": 10, "2": 4,......}
    """
    seen = set()
    repeated = set()
    for lis in lists:
        for data in set(lis):
            if data in seen:
                repeated.add(data)
            else:
                seen.add(data)
    dic = {}
    for num in repeated:
        dic.update({str(num): 0})
    for num in repeated:
        total_num = 0
        for data in lists:
            if num in data:
                total_num += 1
        dic.update({str(num): total_num})
    return dic

y=[3.6214,
3.4848,
3.6118,
]
x=[0,1,2]
f1=interp1d(x, y)
xnew=np.linspace(1, 2, 7)  #插值点
y1=f1(xnew)
print(y1)
