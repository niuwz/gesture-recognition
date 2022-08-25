# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from skimage import feature as sf


def thre(src):
    '''二值图像获取'''
    # OTSU阈值分割
    t, img = cv.threshold(src, 20, 255, cv.THRESH_OTSU)
    # 对二值图像进行去噪操作
    # 开运算
    kernel = np.ones((3, 3), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=3)
    # 中值滤波
    img = cv.medianBlur(img, 7)
    return img


def circle_similar(src):
    '''似圆度'''
    contour, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    length = 0
    area = 0
    # 寻找最大图形的周长和面积
    for i in range(len(contour)):
        c = contour[i].astype(np.float32)
        length = max(length, cv.arcLength(c, True))
        area = max(area, cv.contourArea(c))
    return length**2 / area


def Area(src):
    '''面积'''
    contour, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area = 0
    # 寻找最大图形的面积
    for i in range(len(contour)):
        c = contour[i].astype(np.float32)
        area = max(area, cv.contourArea(c))
    return area


def hr(src):
    '''角点数量'''
    # Harris角点检测
    points = cv.cornerHarris(src, 3, 7, 0.05)
    # 统计非0点数量
    ret = np.count_nonzero(points)
    return ret


def eccentricity(src):
    '''离心率'''
    contours, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    nmax = 0
    # 寻找最大轮廓
    for i in range(len(contours)):
        if len(contours[i]) > nmax:
            nmax = i
    # 计算外接椭圆
    ell = cv.fitEllipse(contours[nmax])
    a, b = max(ell[1])/2, min(ell[1])/2
    c = np.sqrt(a**2-b**2)
    return c/a


def hog(src):
    '''HOG'''
    # 降低图形分辨率以减小特征维度
    img = cv.resize(src, (150, 150))
    hog_array = sf.hog(img, orientations=8, pixels_per_cell=(
        15, 15), cells_per_block=(10, 10))
    return hog_array


def get_features(src):
    '''获取特征向量'''
    # 转换颜色空间
    img = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv.split(img)
    # 对h通道进行阈值分割
    img = thre(h_channel)
    # 获取特征
    a = Area(img)
    s = circle_similar(img)
    e = eccentricity(img)
    n = hr(img)
    h = hog(img)
    return a, s, e, n, h


def get_label(idx):
    '''根据索引获取标签内容'''
    labels = ["paper", "scissors", "rock"]
    return labels[int(idx)]


def get_idx(label):
    '''根据标签获取对应索引'''
    labels = {"paper": 0, "scissors": 1, "rock": 2}
    return labels[label]
