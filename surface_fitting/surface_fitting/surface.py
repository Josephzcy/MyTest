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


def error_func(X,y,k):
    error=np.dot(X,k)-z
    return np.dot(np.transpose(error),error)/(2*m)  #矩阵乘法代替点乘

def gradient_function(X,z,k):
    diff=np.dot(X,k)-z
    return (1./m)*np.dot(np.transpose(X),diff)

def gradient_decent(X,z,alpha):
    #给出初始值
    k= np.array([-5,20,-320]).reshape(3, 1)   #参数初始值怎么设置
    print(k.shape)
    count=0
    gradient=gradient_function(X,z,k)
    # print(gradient)
    # print("调试")
    # k=k-alpha*gradient
    # print(k)
    while not np.all(np.absolute(gradient))<=1e-4 : 
        k=k-alpha*gradient
        gradient=gradient_function(X,z,k)
        count=count+1
        print(count)
    return k,count    #返回多个值用逗号隔开



if __name__=="__main__":
    data = pd.read_excel(path)
    examDf = DataFrame(data)
    m=len(examDf)
    print("数据转为为数组并进行矩阵运算处理")
    x0 = np.ones((m, 1))
    x1= np.array(examDf.iloc[:,0]).reshape(m,1)
    y1= np.array(examDf.iloc[:,1]).reshape(m,1)
    X=np.hstack((x0,x1,y1))
    z= np.array(examDf.iloc[:,2]).reshape(m,1)
    # print(X.shape)
    # print(X)

    k=np.array(symbols('k0,k1,k2')).reshape(3,1)
    error=error_func(X,z,k)
    print(error)

    # alpha=0.01
    # optimal_k,count=gradient_decent(X,z,alpha)
    # print('optimal:',optimal_k)
    # print('count:',count)
    

    print("数据可视化")
    fig2 = plt.figure(2)  # 创建一个画布
    ax2 = Axes3D(fig2)
    ax2.scatter(x1, y1, z, c='r',s=8)  # 绘制数据点,颜色是绿色
    ax2.set_zlabel('Z')  # 坐标轴
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_title('J and scatter', color='r')

    fig1 = plt.figure(1)  # 创建一个画布
    ax1 = Axes3D(fig2)
    ax1.scatter(x1, y1, z, c='r',s=8)  # 绘制数据点,颜色是绿色
    ax1.set_zlabel('Z')  # 坐标轴
    ax1.set_ylabel('Y')
    ax1.set_xlabel('X')
    ax1.set_title('J_error', color='r')




    
    plt.show()