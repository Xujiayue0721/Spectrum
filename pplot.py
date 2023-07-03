import matplotlib.pyplot as plt
import pandas as pd

df1=pd.read_excel(r"D:\建模数据\原始吸收光谱\预测数据 - 20230214.xlsx")
yy=df1.columns[47:203].tolist()
print(yy)
x=df1.values[:,47:203]
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(yy,x.T)
plt.title("原始光谱",fontsize='20') #添加标题
plt.xlabel("Wavelength(nm)")
plt.legend()
plt.show()
