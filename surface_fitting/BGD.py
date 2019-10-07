import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from sympy import *

def error_func(X,y,theta):
    error=np.dot(X,theta)-y
    return (1./(2*m))*np.dot(np.transpose(error),error)

def gradient_function(X,y,theta):
    diff=np.dot(X,theta)-y
    return (1./m)*np.dot(np.transpose(X),diff)

def auto_gradient_function(X,y,theta):
    J=error_func(X,y,theta)
    print(J.shape,theta.shape)
    
    dk=diff(J,theta)
    print(dk.shape)
    dk1=diff(J,theta[0])
    dk2=diff(J,theta[1])
    return dk,dk1,dk2

def gradient_decent(X,y,alpha):
    #给出初始值
    theta = np.array([1, 1]).reshape(2, 1)   #参数初始值怎么设置
    count=0
    gradient=gradient_function(X,y,theta)
    while not np.all(np.absolute(gradient))<=1e-5 : 
        theta= theta-alpha*gradient
        gradient=gradient_function(X,y,theta)
        count=count+1
    return theta,count    #返回多个值用逗号隔开

if __name__=="__main__":
    m = 20
    X0 = np.ones((m, 1))
    X1 = np.arange(1, m+1).reshape(m, 1)
    X = np.hstack((X0, X1))
    y = np.array([
        3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
        11, 13, 13, 16, 17, 18, 17, 19, 21
    ]).reshape(m, 1)

    #绘制X.Y散点图
    # fig = plt.figure()
    # ax= fig.add_subplot(111)
    # ax.scatter(X1,y,c = 'r',marker = 'o')
    # ax.set_title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend('x1')
    print("数据准备完毕")

    # alpha=0.01
    # optimal_theta,count=gradient_decent(X,y,alpha)
    
    # print('optimal:',optimal_theta)
    # print('count:',count)
    
    # optimal_error=error_func(X,y, optimal_theta)
    # print('optimal_error:',optimal_error)
    # #在原来散点图上做出直线
    # Y=optimal_theta[0]+optimal_theta[1]*X1
    # plt.plot(X1,Y,c = 'b')
    # k1,k2=symbols('k1,k2')
    #把符号变量元组转化为符号变量数组
    theta_k=np.array(symbols('k1,k2')).reshape(2,1)
    dk,dk1,dk2=auto_gradient_function(X,y,theta_k)
    gradient=gradient_function(X,y,theta_k)
    
    print(gradient)
    print(dk)
    print(dk1)
    print(dk2)
    # print(theta_k)
    # error_func=error_func(X,y,theta_k)
    # print(error_func.shape)
    # print(error_func)

    # print("绘制误差函数图像")
    # fig2= plt.figure(2)  # 创建一个画布
    # ax2= Axes3D(fig2)
    # k1= np.arange(-10, 10, 0.1)  # 起点、终点、步长
    # k2= np.arange(-10, 10, 0.1)
    # k1,k2= np.meshgrid(k1,k2)
    # # z=pow(k1,2)+pow(k2,2)
    # # ax2.plot_surface(k1,k2, z, rstride=2, cstride=2, cmap='spring')
    # # ax2.plot_surface(k1,k2, plot_error_func, rstride=2, cstride=2, cmap='spring')
    # ax2.set_title('plot_error_func')
    # ax2.set_xlabel('k1')
    # ax2.set_ylabel('k2')
    plt.show()
  