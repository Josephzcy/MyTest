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
import os
path="~/surface_data.xlsx"
path_data="./data/"

def func(P,x,y):
    A,B,C=P
    return A*x+B*y+C
def los_func(P,x,y,z):
     return func(P,x,y)-z

def readData(path_data):  # 传入存储的list    
    list_path_data=[]
    if os.path.exists(path_data):
        for ｆdata in os.listdir(path_data):
            pathdata = os.path.join(path_data,fdata)    #当前文件路径
            list_path_data.append(pathdata) 
    return list_path_data

def data_preprocess(data_path):                         # 传入存储的list  
    dataset = [pd.read_csv(data_path[i]) for i in range(len(data_path))]   #DataFrame元素列表
    X=[dataset[i].iloc[:,0] for i in range(len(dataset))]
    Y=[dataset[i].iloc[:,1] for i in range(len(dataset))]
    Z=[dataset[i].iloc[:,2] for i in range(len(dataset))] 
    mod=[dataset[i].iloc[:,3] for i in range(len(dataset))]

    return X,Y,Z,mod,dataset
  
def surface_fitting(X,Y,Z):  
    P0=[1,10,-4] 
    P1=[1,10,-4]
    P=[]
    P.append(P0)    
    P.append(P1)    #初值设置 
    print("曲面拟合")  

    Para= [  leastsq(los_func,P[i],args=(X[i],Y[i],Z[i])) for i in range((len(X))) ]
    #leastsq（）差值函数、参数初始值，输入的数据
    return Para

def Solve_Incident_Angle(X,Y,Z,para,mod):  
    print(para[0][0])
    print(para[1][0])
    
    # t_mod=[np.linalg.norm([3,4], ord=2)]
    # print(t_mod)         #验证二范数公式

    n_mod=[np.linalg.norm(para[i][0], ord=2) for i in range(len(para))]
    #求所有直线的方向向量  方向向量的模等于点到原点的距离 
  
    #求所有直线的方向向量   转化numpy 方便点成
    v_l0=[ np.array( [ X[0].iloc[j],Y[0].iloc[j],Z[0].iloc[j] ] ) for j in range(len(X[0])) ]  
    v_l1=[ np.array( [ X[1].iloc[j],Y[1].iloc[j],Z[1].iloc[j] ] ) for j in range(len(X[1])) ] 
    v_n=[np.array(para[i][0]) for i in range(len(para))]
    
    # print( "v_n:",v_n,v_n[0].shape)

    # print("v_n[0]", v_n[0]) 
    # print("v_l0[0]",v_l0[0]) 
    # print(np.sum(v_n[0]*v_l0[0]),v_n[0]*v_l0[0]) 


    # print(v_l0)
    nl0=[ np.sum(v_n[0]*v_l0[j])  for j in range(len(v_l0)) ] 
    nl1=[ np.sum(v_n[1]*v_l1[j])  for j in range(len(v_l1)) ] 
     
    print( "nl0:",nl0)
    print(len(nl0))
    # np.around(np.degrees(np.arcｓｉｎ(nl0[i]/(n_mod[0]*mod[0][i]))), decimals = 2)
    # np.around(np.degrees(np.arcｓｉｎ(float(nl1[i]/(n_mod[1]*mod[1][i])))), decimals = 2)
    angle0=[ np.around(np.degrees(np.arcｓｉｎ(nl0[i]/(n_mod[0]*mod[0][i]))), decimals = 2) for i in range(len(nl0)) ]
    angle1=[np.around(np.degrees(np.arcｓｉｎ(float(nl1[i]/(n_mod[1]*mod[1][i])))), decimals = 2) for i in range(len(nl1)) ]

    return angle0, angle1
    

   
    # L=[]
    # L.append(L0)
    # L.append(L1)
    #法向量与方向向量的点模
    # nl0=
    # print(L)

    

if __name__=="__main__":
       
    data_path=readData(path_data)
    print( data_path)
   
    X,Y,Z,mod,dataset=data_preprocess(data_path)
    Para_list=surface_fitting(X,Y,Z)
    Solve_Incident_Angle(X,Y,Z,Para_list,mod)
    
    angle0,angle1=Solve_Incident_Angle(X,Y,Z,Para_list,mod)
    print("angle0",angle0)
    print("angle1",angle1)
   
    
    dataset[0].insert(7,'angle_accumate',angle0)
    dataset[1].insert(7,'angle_accumate',angle1)

    print("存储计算角度数据")
    dataset[0].to_excel("./data/fourth_angle0.xlsx")
    dataset[1].to_excel("./data/fourth_angle1.xlsx")
    
    # print("数据处理完毕")
    


    # fig2 = plt.figure(2)  # 创建一个画布
    # ax2 = Axes3D(fig2)
    # ax2.scatter(x, y, z, c='r')  # 绘制数据点,颜色是绿色
    # ax2.set_zlabel('Z')  # 坐标轴
    # ax2.set_ylabel('Y')
    # ax2.set_xlabel('X')
    # ax2.set_title('J and scatter', color='r')
    
    # print("曲面拟合")
    # P=[5,2,10]                               #初值设置
    # Para=leastsq(los_func,P,args=(x,y,z))   #leastsq（）差值函数、参数初始值，输入的数据
    # print("平面参数值：",Para[0])
    # A,B,C=Para[0]

    # n=[A,B,C]            #平面法向量和模 注意引入了符号变量　    
    # n_r=np.array([A,B,C])
    # n_mod=sqrt(A**2+A**2+C**2)    

    # print("直线值:",x_average,y_average,z_average)
    # ｌ=[x_average,y_average,z_average]    #直线的方向向量和模
    # l_r=np.array([x_average,y_average,z_average] )   #a.tolist()
    # ｌ_mod=sqrt(x_average**2+y_average**2+z_average**2) 
    # print(ｌ_mod)
    # #列表对应元素相乘    　　       
    
    
    # nl=np.sum([a*b for a,b in zip(n,l)])
    # alpha=np.degrees(np.arcｓｉｎ(float(nl/(n_mod*l_mod)))) 
    # print(alpha)
    # nl_r=np.sum(n_r*l_r) 
    # alpha_r=np.degrees(np.arcsin(float(nl_r/(n_mod*l_mod))))
    # print("alpha_r :",alpha_r)

  

    # print("绘制拟合的平面")
    # x, y = np.meshgrid(x, y)
    # J=A*x+B*y+C
    # ax2.plot_surface(x, y, J, rstride=1, cstride=1, cmap='Greens')
    # ax2.plot(x_l, y_l, z_l,c='b')

    # plt.show()

   
   