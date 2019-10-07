import os
import string
import cv2
import numpy as np
import sys
import zlib
import argparse

WIDTH=1920
HEIGHT=1080
DrawBB=True
DrawCULLINGBB=True
Draw3DBB=False
SEGMENTATION=True
COLOR=True
BaseColor=True
DEPTH=True
INSTANCE=True
COMPRESSEDRAW=True
VisualFlow=True
SUBFOLDER=""
import time


def SaveBytearrayToPNG(BinaryData, DstfilePath, width, height):
    
    k = np.zeros((1080,1920,3),np.uint8)
    num = 0
    k = np.frombuffer(BinaryData, dtype = np.uint8)
    k = k.reshape(1080,1920,3)

    cv2.imwrite(DstfilePath+".png", k)
    
def SaveColorToPNG(SrcfilePath, DstfilePath):
    
    f = open(SrcfilePath,'rb')
    filedata = f.read()
    filesize = f.tell()
    f.close()
    BinaryData = bytearray(filedata)
    SaveBytearrayToPNG(BinaryData, DstfilePath, WIDTH, HEIGHT)

def SaveDepthToPNG(SrcfilePath, DstfilePath):
    
    floatData = np.fromfile(SrcfilePath, np.float32)
    k = np.frombuffer(floatData, dtype=np.float32)
    k = k.reshape(1080, 1920)
    cv2.imwrite(DstfilePath + ".png", k)

SegmentationColorTable = [
        [0, 0, 0],
        [107, 142, 35],
        [70, 70, 70],
        [128, 64, 128],
        [220, 20, 60],
        [153, 153, 153],
        [0, 0, 142],
        [0, 0, 0],
        [119, 11, 32],
        [190, 153, 153],
        [70, 130, 180],
        [244, 35, 232],
        [240, 240, 240],
        [220, 220, 0],
        [102, 102, 156],
        [250, 170, 30],
        [152, 251, 152],
        [255, 0, 0],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [111, 74, 0],
        [180, 165, 180],
        [81, 0, 81],
        [150, 100, 100],
        [220, 220, 0],
        [0, 0, 0],
        [250, 170, 160],
        [230, 150, 140],
        [150, 120, 90],
        [0, 0, 0]
    ]
def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.int64)#uint8
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def SegmentationTypeToColor(type):
    if type >= 0 and type < 32:
        return SegmentationColorTable[type]
    return SegmentationColorTable[0]
    
def SaveCompressedSegmentationToPNG(SrcfilePath, DstfilePath, DstfilePath1):
   
   
    f = open(SrcfilePath,'rb')
    zlibdata = f.read()
    f.close()
    try:
        filedata = zlib.decompress(zlibdata)
    except:
        filedata = zlibdata
    BinaryData = bytearray(filedata)
   
   
    #k = np.zeros((1080,1920),np.int8)
    #num=0
    k = np.frombuffer(BinaryData, dtype=np.uint8)
    k = k.reshape(1080, 1920)
    pred_color = colorEncode(k, SegmentationColorTable)

    cv2.imwrite(DstfilePath + ".png", k)
    cv2.imwrite(DstfilePath1 + ".png", pred_color)

def SaveCompressedInstanceToPNG(SrcfilePath, DstfilePath):

    f = open(SrcfilePath,'rb')
    zlibdata = f.read()
    f.close()
    try:
        filedata = zlib.decompress(zlibdata)
    except:
        filedata = zlibdata

    BinaryData = bytearray(filedata)
    SaveBytearrayToPNG(BinaryData, DstfilePath, WIDTH, HEIGHT)
   
FlowFlagBitTable = [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0]
    ]

def SaveFlowFlagBitToPNG(SrcfilePath, DstfilePath):
   
   
    f = open(SrcfilePath,'rb')
    filedata = f.read()
    f.close()
    BinaryData = bytearray(filedata)
    k = np.frombuffer(BinaryData, dtype=np.uint8)
    k = k.reshape(1080, 1920)
    pred_color = colorEncode(k, FlowFlagBitTable)

    cv2.imwrite(DstfilePath + ".png", pred_color)   
    
def DestPath(inPath):
    
    dstPath = inPath 
    if os.path.exists(dstPath):
         os.remove(dstPath)
    return dstPath
    
def gci(filepath,args):

  files = os.listdir(filepath)

  for fi in files:
    fi_d = os.path.join(filepath,fi)
    if not os.path.isdir(fi_d):
        continue
    files1 = os.listdir(fi_d)
    print(fi_d)
    for fi_z in files1:
        fi_k = os.path.join(fi_d,fi_z)
        if not os.path.isfile(fi_k):
            continue

        if fi_z == "Color" and COLOR:
            Color_path = args.output + '/Color/'
            if not os.path.exists(Color_path):
                os.mkdir(Color_path)
            SaveColorToPNG(fi_k, DestPath(Color_path + 'Color' + fi))
        elif fi_z == "DepthPlanner" and DEPTH:
            DepthPlanner_path = args.output + '/DepthPlanner/'
            if not os.path.exists(DepthPlanner_path):
                os.mkdir(DepthPlanner_path)
            SaveDepthToPNG(fi_k, DestPath(DepthPlanner_path + 'DepthPlanner' + fi))
        elif fi_z == "Segmentation" and SEGMENTATION:
            Segmentation_path = args.output + '/Segmentation/'
            color_Segmentation_path = args.output + '/SegmentationView/'

            if not os.path.exists(Segmentation_path):
                os.mkdir(Segmentation_path)
            if not os.path.exists(color_Segmentation_path):
                os.mkdir(color_Segmentation_path)
            SaveCompressedSegmentationToPNG(fi_k, DestPath(Segmentation_path + 'Segmentation' + fi), DestPath(
                color_Segmentation_path + 'Segmentation_View' + fi))  # "/home/wj/桌面/sege/Segmentation"
        elif fi_z == "Instance" and INSTANCE:
            Instance_path = args.output + '/Instance/'
            if not os.path.exists(Instance_path):
                os.mkdir(Instance_path)
            SaveCompressedInstanceToPNG(fi_k, DestPath(Instance_path + 'Instance' + fi))
        if fi_z == "BaseColor" and BaseColor:
            BaseColor_path = args.output + '/BaseColor/'
            if not os.path.exists(BaseColor_path):
                os.mkdir(BaseColor_path)
            SaveColorToPNG(fi_k, DestPath(BaseColor_path + 'BaseColor' + fi))
        if fi_z == "VisualFlow" and VisualFlow:
            VisualBackFlow_path = args.output + '/VisualFlow/'
            if not os.path.exists(VisualBackFlow_path):
                os.mkdir(VisualBackFlow_path)
            SaveColorToPNG(fi_k, DestPath(VisualBackFlow_path + 'VisualFlow' + fi))
        if fi_z == "VisualFlowForward" and VisualFlow:
            VisualFlowForward_path = args.output + '/VisualFlowForward/'
            if not os.path.exists(VisualFlowForward_path):
                os.mkdir(VisualFlowForward_path)
            SaveColorToPNG(fi_k, DestPath(VisualFlowForward_path + 'VisualFlowForward' + fi))
        if fi_z == "FlowFlagBit":
            FlowFlagBit_path = args.output + '/FlowFlagBit/'
            if not os.path.exists(FlowFlagBit_path):
                os.mkdir(FlowFlagBit_path)
            SaveFlowFlagBitToPNG(fi_k, DestPath(FlowFlagBit_path + 'FlowFlagBit' + fi))

 
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    # Model related arguments
    print('Usage: RAW2PNG.py --input=inputpath --output=outpath')
    parser.add_argument('--input', default='./', required=False,
                        help="a path input file")
    parser.add_argument('--output', default='./out', required=False,
                        help="a path out file")
  
    args = parser.parse_args()
    CurworkingDir=os.path.realpath(args.input)

    if len(SUBFOLDER)>0:
        CurworkingDir=os.path.join(CurworkingDir,SUBFOLDER)

    if not os.path.exists(args.output):
           os.mkdir(args.output)

    print('start converting........')
    start_time = time.time()
    gci(CurworkingDir,args)
    print('over.........',time.time()-start_time)

