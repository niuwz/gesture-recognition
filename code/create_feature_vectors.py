# coding:utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from skimage import feature as sf
from gesture_recognize_utils import *

if __name__ == "__main__":
    path = "gesture_recognize/data"
    types = ["paper", "rock", "scissors"]

    areas = []
    css = []
    es = []
    ns = []
    hs = []
    labels = []
    # 对不同类别中的每一个图像提取特征
    # 遍历文件夹中数据，根据父文件夹名称获取标签
    for label in types:
        pathi = path+"/"+label
        names = os.listdir(pathi)
        for name in names:
            filename = pathi+"/"+name
            src = cv.imread(filename)
            a, s, e, n, h = get_features(src)
            areas.append(a)
            es.append(e)
            css.append(s)
            ns.append(n)
            hs.append(h)
            labels.append(label)
    # 将提取到的特征向量转换成pd.DataFrame后保存为csv
    data = {
        u"面积": areas,
        u"离心率": es,
        u"似圆度": css,
        u"角点数量": ns,
    }
    hs = np.array(hs).reshape(len(hs), -1)
    label_dict = {u"类别": labels}

    df = pd.DataFrame(data)
    hs_df = pd.DataFrame(hs)
    label_df = pd.DataFrame(label_dict)
    df = pd.concat([df, hs_df, label_df], axis=1)
    # 注意csv需要使用gbk编码
    df.to_csv("gesture_recognize/data/features.csv", index=0, encoding="gbk")
