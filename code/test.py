import os
import warnings

import cv2 as cv
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm

from gesture_recognize_utils import *

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    file_path = "gesture_recognize/data/nwz"
    files = os.listdir(file_path)
    # 加载训练好的SVM分类器
    svmClassfication: svm = joblib.load("gesture_recognize/svm.pkl")
    cnt = 0
    print("分类错误样本:")
    for i in range(len(files)):
        name = files[i]
        file_name = file_path+"/"+name
        # 分类文件名中的标签信息
        label = name.split("_")[1]
        # 转换为索引以便于比较
        label_idx = get_idx(label)
        # 读取图像并提取特征
        src = cv.imread(file_name)
        a, s, e, n, h = get_features(src)
        x = np.array([a, s, e, n])
        x = np.hstack([x, h]).reshape(1, -1)
        # 获取预测值(标签索引)
        y = svmClassfication.predict(x)
        # 判断预测的准确性并计算准确率
        if label_idx == y.item():
            cnt += 1
        else:
            print("样本{}:标签{},被错分为:{}".format(i+1, label, get_label(y.item())))
    if cnt == len(files):
        print("无")
    print(f"\n分类准确率:{cnt/len(files)*100}%")
    joblib.dump(svmClassfication, 'gesture_recognize/svm.pkl')
