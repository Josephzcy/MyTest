import pandas as pd
from pandas import DataFrame,Series
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy.optimize import leastsq
from scipy.optimize import root,fsolve
from mpl_toolkits.mplot3d import Axes3D    #Matplotlib里面专门用来画三维图的工具包
import  seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import train_test_split
path="./surface_data.xlsx"
print(path)

def func(P,x,y):
    A,B,C=P
    return A*x+B*y+C
def los_func(P,x,y,z,s):
     print(s)
     return func(P,x,y)-z
if __name__=="__main__":
    data = pd.read_excel(path)
    examDf = DataFrame(data)
    surface_fitting= DataFrame(columns=['x', 'y', 'z'])
    surface_fitting['x'] = examDf.iloc[:,0]
    surface_fitting['y'] = examDf.iloc[:,1]
    surface_fitting['z'] = examDf.iloc[:,2]
    print(surface_fitting)
    print("数据处理完毕")
   
    print("数据可视化")
    x=surface_fitting['x']
    y=surface_fitting['y']
    z=surface_fitting['z']

    fig2 = plt.figure(2)  # 创建一个画布
    ax2 = Axes3D(fig2)
    ax2.scatter(x, y, z, c='r')  # 绘制数据点,颜色是绿色
    ax2.set_zlabel('Z')  # 坐标轴
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    
    print("曲面拟合")
    s="Test the number of iteration" 
    P=[5,2,10]
    Para=leastsq(los_func,P,args=(x,y,z,s))
    print(Para[0])
    A,B,C=Para[0]
    
    print("绘制拟合函数图像")
    fig = plt.figure(1)  # 创建一个画布
    ax1 = Axes3D(fig)
    x, y = np.meshgrid(x, y)
    J=A*x+B*y+C
    ax1.plot_surface(x, y, J, rstride=1, cstride=1, cmap='rainbow')
    ax1.set_xlabel('x ', color='r')
    ax1.set_ylabel('y ', color='g')
    ax1.set_zlabel('J ', color='b')  # 给三个坐标轴注明
    ax1.set_title('J', color='r')
    plt.show()

    # 需要优化：将原来的点在拟合的平面中标出来
   