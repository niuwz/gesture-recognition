# coding:utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import model_selection

import joblib
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # 读取特征向量数据集
    data = pd.read_csv(
        "gesture_recognize/data/features.csv", encoding="gbk")
    # 将类别转换为索引
    data["类别"][data["类别"] == "paper"] = 0
    data["类别"][data["类别"] == "scissors"] = 1
    data["类别"][data["类别"] == "rock"] = 2
    x_data = data.iloc[:, :-1].to_numpy(np.float32)
    y_data = data["类别"].to_numpy(dtype=np.float32)
    # 分割训练集和测试集
    train_data, test_data, train_label, test_label = model_selection.train_test_split(
        x_data, y_data, random_state=1, test_size=0.25, train_size=0.75)
    # 定义SVM分类器
    svmClassfication = svm.SVC(
        C=100, kernel="rbf", gamma=1e-5, decision_function_shape="ovr", max_iter=50000)
    # 训练SVM分类器
    svmClassfication = svmClassfication.fit(train_data, train_label)
    # 测试分类器在训练集和验证集上的结果
    print("训练集:{: >8.4f}%".format(
        svmClassfication.score(train_data, train_label)*100))
    print("验证集:{: >8.4f}%".format(
        svmClassfication.score(test_data, test_label)*100))
    # 保存分类器
    joblib.dump(svmClassfication, 'gesture_recognize/svm.pkl')
