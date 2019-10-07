import numpy as np
import cv2 as cv
import time as t
from numba import jit
from itertools import product
start_time=t.process_time()
path='C:\Dump_light_stream\OpticalFlow-190705-142119\\1813\\FlowGroundTruth'
path_image_1813='C:\Dump_light_stream\OpticalFlow-190705-142119\out\BaseColor\BaseColor1813.png'
path_image_1801='C:\Dump_light_stream\OpticalFlow-190705-142119\out\BaseColor\BaseColor1801.png'
save_path='F:\Pycharm_docments\Channel_flow.txt'
flow_img_3C=np.zeros((1080,1920,1))
Channel_flow=np.zeros((1080,1920),np.dtype(int))
visualization_Channel_flow=np.zeros((1080,1920,3))

def get_flow_img_np(in_file, height, width):         # height：y轴长度；width x轴长度
    flow_img = np.fromfile(in_file, np.float32)
    assert len(flow_img.shape) == 1
    assert flow_img.shape[0] == height * width * 2
    flow_img = np.reshape(flow_img, (height, width, 2))
    return flow_img

#读取原图像
src_1813=cv.imread(path_image_1813)
# cv.namedWindow("1813_visualization",cv.WINDOW_NORMAL)
# cv.imshow("data1813_visualization", src_1813)

src_1801=cv.imread(path_image_1801)

# src_1801_copy = np.array(src_1801)

# cv.namedWindow("1801_visualization",cv.WINDOW_NORMAL)
# cv.imshow("1801_visualization",src_1801)

# print('流数据处理')
flow_img=get_flow_img_np(path, 1080, 1920)
# print('给流数据增加一个通道')
# flow_img = np.c_[flow_img,np.zeros((1080,1920,1))]
# cv.imwrite('F:\Pycharm_docments\flow_img.txt',flow_img)
#print(flow_img)  #2个通道
#cv.namedWindow("flow_visualization",cv.WINDOW_NORMAL)
#cv.imshow('flow_visualization', flow_img)
print('开始匹配处理')
start_time1=t.process_time()
# @jit
@jit
for row in range(1,src_1813.shape[0]-1):  # 遍历高     1080
    for col in range(1,src_1813.shape[1]-1):  # 遍历宽  1920
        for c in range(src_1813.shape[2]):
            x_d = row + int(flow_img[row, col, 0])  # x_d 表示位于那一行 ，表示横坐标
            y_d = col + int(flow_img[row, col, 1])  # y_d 表示位于那一列 ，表示纵坐标        (src_1801[row + 1, col + 1,c] - src_1813[x_d, y_d,c]))
            if 0 < x_d < 1080 and 0 < y_d < 1920:  # 未超出边界的点要确定是正常还是遮挡
                if abs( int(src_1801[row -1,col-1,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row - 1,col,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row - 1, col + 1,c])-int(src_1813[x_d, y_d,c])) < 25 or \
                   abs( int(src_1801[row,   col-1,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row,    col,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row,     col + 1,c])-int(src_1813[x_d, y_d,c]) )< 25 or \
                   abs( int(src_1801[row +1,col-1,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row + 1,col,c])-int(src_1813[x_d, y_d,c])) < 25 or abs( int(src_1801[row + 1, col + 1,c])-int(src_1813[x_d, y_d,c]))< 25 :
                        # print('正确匹配到')
                            Channel_flow[row, col] = 0
                        #deviation_flow[row, col] = src_1813[x_d, y_d]   #第一个下标不能超过1080 第二个坐标不能超过1920

                        # src_1801_copy[x_d, y_d] = [255, 0, 0]
                        # print('1801修改之后')
                        #找到了之后1801像素值不变
                else:
                        # print('被遮挡')
                         Channel_flow[row, col] = 2
                        # src_1801_copy[x_d, y_d] = [0, 0, 255]
                        # 遮挡了用红色或者用遮挡之后的像素值
                        # src_1801_copy[row, col] = [0, 0, 255]
                        #src_1801_copy[row, col] = src_1813[x_d, y_d]
            else:
                  # print('该点已超出边界')
                  Channel_flow[row, col] = 1
                  #src_1801_copy[row, col] = [0, 255,0]
                  # src_1801_copy[row, col] = [0, 255, 0]


start_time2=t.process_time()
print(start_time2-start_time1)

start_time3=t.process_time()


# print(' 显示流通道数据')
# print(Channel_flow,Channel_flow.shape)
# print(' 可视化流通道数据')
print('开始可视化处理')
for row in range(Channel_flow.shape[0]):  # 遍历高1080
    for col in range(Channel_flow.shape[1]):  # 遍历宽
         if  Channel_flow[row,col]==0 :    #正常
                visualization_Channel_flow[row,col]=[255,0,0]
         if  Channel_flow[row,col]==1 :    #绿色代表跑出去了
                visualization_Channel_flow[row,col]=[0,255,0]
         if Channel_flow[row, col] == 2:   #红色代表被遮挡了
                visualization_Channel_flow[row, col]= [0, 0, 255]
start_time4=t.process_time()
print(start_time4-start_time3)
# np.savetxt(save_path, Channel_flow,fmt='%d')
cv.namedWindow("visualization_Channel_flow",cv.WINDOW_NORMAL)
cv.imshow('visualization_Channel_flow', visualization_Channel_flow)

# cv.namedWindow("visualization_modify_1801",cv.WINDOW_NORMAL)
# cv.imshow('visualization_modify_1801', src_1801_copy)


# print('左上角处理')       #(0,0)
# # 左上角处理(0,1919)
# if  0<=0+int(flow_img[0, 0, 0])<1080 and 0 <0+int(flow_img[0, 0, 1] )< 1920:
#     x0_d = (0 + int(flow_img[0,0,0]))  # x_d 表示位于那一行 ，表示横坐标
#     y0_d = (0 + int(flow_img[0,0,1]))  # y_d 表示位于那一列 ，表示纵坐标
#     if ((src_1801[0,0] - src_1813[x0_d, y0_d]) < 10).any() or ((src_1801[0,1] - src_1813[x0_d, y0_d]) < 10).any()  or \
#        ((src_1801[1,0] - src_1813[x0_d, y0_d]) < 10).any() or ((src_1801[1,1] - src_1813[x0_d, y0_d]) < 10).any() or \
#        ((src_1801[2,0] - src_1813[x0_d, y0_d]) < 10).any() or ((src_1801[2,1] - src_1813[x0_d, y0_d]) < 10).any() :
#                 Channel_flow[0, 0] = 0  # 正确找到
#     else :
#                 Channel_flow[0, 0] = 2   # 被遮挡  [0,0,0]
#                 src_1801[0, 0] = [0, 0, 255]
#                 src_1801_copy[0, 0] = src_1813[x0_d, y0_d]
# else :
#         Channel_flow[0, 0] = 1 # 被遮挡
#         src_1801[0, 0] = [0, 255, 0]
#         src_1801_copy[0, 0] = [0, 255, 0]
#
# # 右上角处理(0,1919)
# print('右上角处理')
# if  0<=0+int(flow_img[0, 1919, 0])<1080 and 0 <1919+int(flow_img[0, 1919, 1] )< 1920:
#     x0_d    = 0 + int(flow_img[0,1919,0])  # x_d 表示位于那一行 ，表示横坐标
#     y1919_d = 1919 + int(flow_img[0,1919,1])  # y_d 表示位于那一列 ，表示纵坐标
#     if ((src_1801[0,1918] - src_1813[x0_d, y1919_d]) < 10).any() or ((src_1801[0,1919] - src_1813[x0_d, y1919_d]) < 10).any()  or\
#        ((src_1801[1,1918] - src_1813[x0_d, y1919_d]) < 10).any() or ((src_1801[1,1919] - src_1813[x0_d, y1919_d]) < 10).any()  :
#                 Channel_flow[0, 1919] = 0  # 正确找到
#     else :
#                 Channel_flow[0, 1919] = 2   # 被遮挡
#                 src_1801[0, 1919] = [0, 0, 255]
#                 src_1801_copy[0, 1919] = src_1813[x0_d, y1919_d]
# else :
#       Channel_flow[0, 1919] = 1 # 跑出去     [0,1919,0]
#       src_1801[0, 1919] = [0, 255, 0]
#       src_1801_copy[0, 1919] = [0, 255, 0]
#
#
# print('左下角处理')
# # 左下角处理(1079,0)
# if  0<=1079+int(flow_img[1079, 0, 0])<1080 and 0 <0+int(flow_img[1079, 0, 1] )< 1920:
#     x1079_d = 1709 + int(flow_img[1079,0,0])  # x_d 表示位于那一行 ，表示横坐标
#     y0_d    = 0 + int(flow_img[1079,0,1])  # y_d 表示位于那一列 ，表示纵坐标
#     if ((src_1801[1078,0] - src_1813[x1079_d, y0_d]) < 10).any() or ((src_1801[1078,1] - src_1813[x1079_d, y0_d]) < 10).any() or \
#        ((src_1801[1079,0] - src_1813[x1079_d, y0_d]) < 10).any() or ((src_1801[1079,1] - src_1813[x1079_d, y0_d]) < 10).any() :
#                 Channel_flow[1079, 0] = 0  # 正确找到
#     else :
#                 Channel_flow[1079, 0] = 2   # 被遮挡
#                 src_1801[1079, 0] = [0, 0, 255]
#                 src_1801_copy[1079, 0] = src_1813[x1079_d, 0]
#
# else :
#      Channel_flow[1079, 0] = 1 # 被遮挡   [1079,0,0]
#      src_1801[1079, 0] = [0, 255, 0]
#      src_1801_copy[1079, 0] = [0, 255, 0]
# #
# print('右下角处理')
# # 右下角处理(1079,1919)
# if 0 <= 1079 + int(flow_img[1079, 0, 0]) < 1080 and 0 < 1919+ int(flow_img[1919, 0, 1]) < 1920:
#     x1079_d = 1709+ int(flow_img[1079, 1919,0])  # x_d 表示位于那一行 ，表示横坐标
#     y1919_d = 1919+ int(flow_img[1079, 1919,1])  # y_d 表示位于那一列 ，表示纵坐标
#     if ((src_1801[1078, 1918] - src_1813[x1079_d, y1919_d]) < 10).any() or ((src_1801[1078, 1919] - src_1813[x1079_d, y1919_d]) < 10).any() or\
#        ((src_1801[1079, 1918] - src_1813[x1079_d, y1919_d]) < 10).any() or ((src_1801[1079, 1919] - src_1813[x1079_d, y1919_d]) < 10).any():
#             Channel_flow[1079, 1919] = 0  # 正确找到
#     else:
#             Channel_flow[1079, 1919] = 2  # 被遮挡  [1079, 1919,0]
#             src_1801[1079, 1919] = [0, 0, 255]
#             src_1801_copy[1079, 1979] = src_1813[x1079_d,  y1919_d]
# else:
#      Channel_flow[1079, 1919] = 1  # 被遮挡
#      src_1801[1079, 1919] = [0, 255, 0]
#      src_1801_copy[1079,1919] = [0, 255, 0]
#
# print('第一行边界处理')   #比较6个点(0,col)
# for col in range(1,src_1813.shape[1]-1):  # 从第二列开始到1819列
#     if 0 <= 0 + int(flow_img[0, col, 0]) < 1080 and 0 < col + int(flow_img[0, col, 1]) < 1920:
#         x0_d = 0 + int(flow_img[0, col, 0])  # x_d 表示位于那一行 ，表示横坐标
#         ycol_d = col + int(flow_img[0, col, 1])  # y_d 表示位于那一列 ，表示纵坐标
#         if ((src_1801[0, col-1] - src_1813[x0_d, ycol_d]) < 10).any() or ((src_1801[0, col] - src_1813[x0_d, ycol_d]) < 10).any() or ((src_1801[0, col+1] - src_1813[x0_d, ycol_d]) < 10).any() or\
#           ((src_1801 [1, col-1] - src_1813[x0_d, ycol_d]) < 10).any() or ((src_1801[1, col] - src_1813[x0_d, ycol_d]) < 10).any()  or ((src_1801[1, col+1] - src_1813[x0_d, ycol_d]) < 10).any():
#                 Channel_flow[0, col] = 0  # 正确找到
#         else:
#                 Channel_flow[0, col] = 2  # 被遮挡
#                 src_1801[0, col] = [0, 0, 255]
#                 src_1801_copy[0, col] = src_1813[x0_d, ycol_d]
#     else:
#         Channel_flow[0, col] = 1  # 被遮挡
#         src_1801[0, col] = [0, 255, 0]
#         src_1801_copy[0, col] = [0, 255, 0]
#
# print('最后一行边界处理')   #比较6个点(1079,col)
# for col in range(1,src_1813.shape[1]-1):  # 遍历宽1920-1
#     if 0 <= 1079 + int(flow_img[1079, 0, 0]) < 1080 and 0 < col + int(flow_img[1079, 0, 1]) < 1920:
#         x1079_d = 1079 + int(flow_img[1079, col, 0])  # x_d 表示位于那一行 ，表示横坐标
#         ycol_d = col + int(flow_img[1079, col, 1])  # y_d 表示位于那一列 ，表示纵坐标
#         if ((src_1801[1078, col - 1] - src_1813[x1079_d, ycol_d]) < 10).any() or ((src_1801[1078, col] - src_1813[x1079_d, ycol_d]) < 10).any() or ((src_1801[1078, col + 1] - src_1813[x1079_d, ycol_d]) < 10).any() or\
#            ((src_1801[1079, col - 1] - src_1813[x1079_d, ycol_d]) < 10).any() or ((src_1801[1079, col] - src_1813[x1079_d, ycol_d]) < 10).any() or ((src_1801[1079, col + 1] - src_1813[x1079_d, ycol_d]) < 10).any():
#                 Channel_flow[1079, col] = 0  # 正确找到
#         else:
#                 Channel_flow[1079, col] = 2  # 被遮挡
#                 src_1801[1079, col] = [0, 0, 255]
#                 src_1801_copy[1079, col] = src_1813[x1079_d, ycol_d]
#     else:
#         Channel_flow[1079, col] = 1  # 被遮挡
#         src_1801[1079, col] = [0, 255, 0]
#         src_1801_copy[1079, col] = [0, 255, 0]
#
# print('第一列边界处理')  #坐标(row,0)
# for row in range(1, src_1813.shape[0]-1):  # 遍历宽
#     if 0 <= row+ int(flow_img[row, 0, 0]) < 1080 and 0 < 0 + int(flow_img[row, 0, 1]) < 1920:
#         xrow_d = row+ int(flow_img[row, 0, 0])  # x_d 表示位于那一行 ，表示横坐标
#         y0_d = 0 + int(flow_img[row, 0, 1])  # y_d 表示位于那一列 ，表示纵坐标
#         if ((src_1801[xrow_d-1, 0] - src_1813[xrow_d, y0_d]) < 10).any() or ((src_1801[ xrow_d-1 ,       1] - src_1813[xrow_d, y0_d]) < 10).any() or \
#            ((src_1801[xrow_d,   0] - src_1813[xrow_d, y0_d]) < 10).any() or ((src_1801[ xrow_d ,         1] - src_1813[xrow_d, y0_d]) < 10).any() or \
#            ((src_1801[xrow_d+1 ,0] - src_1813[xrow_d, y0_d]) < 10).any() or ((src_1801[ xrow_d +1, col + 1] - src_1813[xrow_d, y0_d]) < 10).any():
#                 Channel_flow[row, 0] = 0  # 正确找到
#         else:
#                 Channel_flow[row, 0] = 2  # 被遮挡
#                 src_1801[row, 0] = [0, 0, 255]
#                 src_1801_copy[row, 0] = src_1813[xrow_d, y0_d]
#     else:
#             Channel_flow[row, 0] = 1  # 被遮挡
#             src_1801[row, 0] = [0, 255, 0]
#             src_1801_copy[row, 0] = [0, 255, 0]
# #
#
# print('最后一列边界处理')  #(row,1919)
# for row in range(1, src_1813.shape[0]-1):  # 遍历宽
#     if 0 <= row+ int(flow_img[row, 0, 0]) < 1080 and 0 < 1919 + int(flow_img[row, 0, 1]) < 1920:
#         xrow_d = row+ int(flow_img[row, 0, 0])  # x_d 表示位于那一行 ，表示横坐标
#         y1919_d = 0 + int(flow_img[row, 0, 1])  # y_d 表示位于那一列 ，表示纵坐标
#         if ((src_1801[xrow_d-1, 1918] - src_1813[xrow_d, y1919_d]) < 10).any() or ((src_1801[ xrow_d-1 ,1919] - src_1813[xrow_d, y1919_d]) < 10).any() or \
#            ((src_1801[xrow_d,   1918] - src_1813[xrow_d, y1919_d]) < 10).any() or ((src_1801[ xrow_d ,  1919] - src_1813[xrow_d, y1919_d]) < 10).any() or \
#            ((src_1801[xrow_d+1 ,1918] - src_1813[xrow_d, y1919_d]) < 10).any() or ((src_1801[ xrow_d +1,1919] - src_1813[xrow_d, y1919_d]) < 10).any():
#                 Channel_flow[row, 1919] = 0  # 正确找到
#         else:
#                 Channel_flow[row, 1919] = 2  # 被遮挡
#                 src_1801[row, 1919] = [0, 0, 255]
#                 src_1801_copy[row, 1919] = src_1813[xrow_d, y1919_d]
#     else:
#             Channel_flow[row, 1919] = 1  # 被遮挡
#             src_1801[row, 1919] = [0, 255, 0]
#             src_1801_copy[row, 1919] = [0, 255, 0]

# print(' 显示流通道数据')
# print(Channel_flow,Channel_flow.shape)
# print(' 可视化流通道数据')
# for row in range(Channel_flow.shape[0]):  # 遍历高1080
#     for col in range(Channel_flow.shape[1]):  # 遍历宽
#          if  Channel_flow[row,col]==0 :    #正常
#                 visualization_Channel_flow[row,col]=[255,0,0]
#          if  Channel_flow[row,col]==1 :    #绿色代表跑出去了
#                 visualization_Channel_flow[row,col]=[0,255,0]
#          if Channel_flow[row, col] == 2:   #红色代表被遮挡了
#                 visualization_Channel_flow[row, col]= [0, 0, 255]

# np.savetxt(save_path, Channel_flow,fmt='%d')
# cv.namedWindow("visualization_Channel_flow",cv.WINDOW_NORMAL)
# cv.imshow('visualization_Channel_flow', visualization_Channel_flow)

#显示修改之后的1813

# cv.namedWindow("visualization_modify_1801_copy",cv.WINDOW_NORMAL)
# cv.imshow('visualization_modify_1801_copy', src_1801_copy)

# end_time = t.process_time()
# run_time = end_time - start_time
# print(run_time)
cv.waitKey(0)








# src_1813.shape[0] = src_1813.shape[0]
# src_1813.shape[1] = src_1813.shape[1]
# src_1813_channels = src_1813.shape[2]
# print("weight : %s, height : %s, channel : %s" % (src_1813.shape[0], src_1813.shape[1], src_1813_channels))
# print('判断像素范围内的点是否能匹配')   #每个点和领域的9个点进行比较
# for row in range(1,src_1813.shape[0]-1):  # 遍历高
#     for col in range(1,src_1813.shape[1]-1):  # 遍历宽
#         if 0 <= row+ int(flow_img[0, 0, 0]) < 1080 and 0 < col+ int(flow_img[0, 0, 1]) < 1920:
#             if ((src_1801[row-1,col-1]- deviation_flow[row,col])<10).any() or  ((src_1801[row-1,col] - deviation_flow[row,col])<10).any() or ((src_1801[row-1,col+1] - deviation_flow[row, col])<10).any() or \
#                ((src_1801[row, col-1] -deviation_flow[row,col])<10).any() or ((src_1801[row,col] -deviation_flow[row, col]<10)).any() or  ((src_1801[row,col+1] - deviation_flow[row, col])<10).any() or \
#                ((src_1801[row+1,col-1] - deviation_flow[row,col])<10).any() or ((src_1801[row+1,col] - deviation_flow[row,col])<10).any() or ((src_1801[row+1,col+1] - deviation_flow[row, col])<10).any() :
#                 #print('正确匹配到')
#                    Channel_flow[row, col] = 0    #正确找到
#             else :
#                    Channel_flow[row, col] = 2    #被挡住了
#     else:
#         print('正常区域内有数据出界')
#         Channel_flow[row, col] = 1

# for row in range(1,src_1813.shape[0]-1):  # 遍历高
#     for col in range(1,src_1813.shape[1]-1):  # 遍历宽
#         if 0 <= row+ int(flow_img[0, 0, 0]) < 1080 and 0 < col+ int(flow_img[0, 0, 1]) < 1920:
#             if (src_1801[row-1,col-1]== deviation_flow[row,col]).any() or  (src_1801[row-1,col] == deviation_flow[row,col]).any() or (src_1801[row-1,col+1] == deviation_flow[row, col]).any() or \
#                (src_1801[row, col-1] == deviation_flow[row,col]).any() or (src_1801[row,col] == deviation_flow[row, col]).any() or  (src_1801[row,col+1] == deviation_flow[row, col]).any() or \
#                (src_1801[row+1,col-1] == deviation_flow[row,col]).any() or (src_1801[row+1,col] == deviation_flow[row,col]).any() or (src_1801[row+1,col+1] == deviation_flow[row, col]).any() :
#                 #print('正确匹配到')
#                    Channel_flow[row, col] = 0    #正确找到
#             else :
#                    Channel_flow[row, col] = 2    #被挡住了
#     else:
#         print('正常区域内有数据出界')
#         Channel_flow[row, col] = 1

# cv.destroyAllWindows()

#将图片转为灰度图
# cropImg_1813_gray = cv.cvtColor(cropImg_1813,cv.COLOR_RGB2GRAY)
# cropImg_1801_gray = cv.cvtColor(cropImg_1801,cv.COLOR_RGB2GRAY)
#循环比较
# print('获取每个像素点的每个通道的数值')
# for row in range(cropImg_1813_height):
#     for col in range(cropImg_1813_Width):
#         for c in range(cropImg_1813_channels):
#               pv = cropImg_1813[row, col, c]
#               print(pv)

# print('判断点是否越界')
# for row in range(src_1813.shape[0]):  # 遍历高     1080
#     for col in range(src_1813.shape[1]):  # 遍历宽  1920
#         x_d = row + int(flow_img[row, col, 0])   #x_d 表示位于那一行 ，表示横坐标
#         y_d= col + int(flow_img[row, col, 1])    #y_d 表示位于那一列 ，表示纵坐标
#         if 0 < x_d < 1080 and 0 <y_d < 1920:
#             # print('ok')
#             deviation_flow[row, col] = src_1813[x_d, y_d]   #第一个下标不能超过1080 第二个坐标不能超过1920
#             Channel_flow[row, col]=0
#         else:
#             # print('该点已超出边界')
#              Channel_flow[row, col] = 1
# cv.namedWindow("deviation_visualization",cv.WINDOW_NORMAL)
# cv.imshow('deviation_visualization', deviation_flow)

# def df_Occlusion_match(img1, img2):         # height：y轴长度；width x轴长度
#     for row in range(1, img1.shape[0] - 1):  # 遍历高
#         for col in range(1,img1.shape[1] - 1):  # 遍历宽
#             # if 0 <= row + int(flow_img[0, 0, 0]) < 1080 and 0 < col + int(flow_img[0, 0, 1]) < 1920:
#                 if ((img1[row - 1, col - 1] - img2[row, col]) < 10).any() or ((img1[row - 1, col] - img2[row, col]) < 10).any() or \
#                   ((img1[row - 1, col + 1] - img2[row, col]) < 10).any() or ((img1[row, col - 1] - img2[row, col]) < 10).any() or \
#                   ((img1[row, col] - img2[row, col] < 10)).any() or ((img1[row, col + 1] - img2[row, col]) < 10).any() or \
#                   ((img1[row + 1, col - 1] - img2[row, col]) < 10).any() or ((img1[row + 1, col] - img2[row, col]) < 10).any() or (
#                  (img1[row + 1, col + 1] - deviation_flow[row, col]) < 10).any():
#                     # print('正确匹配到')
#                         Channel_flow[row, col] = 0  # 正确找到
#                 else:
#                         Channel_flow[row, col] = 2  # 被挡住了
#         # else:
#         #     print('正常区域内有数据出界')
#         #     Channel_flow[row, col] = 1


