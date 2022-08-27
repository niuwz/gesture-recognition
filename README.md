
本文为人工智能课程设计的第一道题目，内容为手势图像(剪刀石头布图像)的识别与分类，收到数据集数量和大小的限制，本文中只提供了若干张图片用于训练，完整的训练数据集下载链接[如下](https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip)。所用到的测试数据为部分自采图像与部分测试图像，这里同样只展示一小部分。

本文中所采用的模型框架如下图所示：

<figure>
<center>
<img src="pictures/svm框架.png">
</center>
</figure>

## 内容简介

```
gesture_recognize_utils中包含了所需的所有函数
```
```
create_feature_vectors用于生成图像的特征向量文件，为features.csv
```
```
classfication中包含了svm分类器的训练和验证
```
```
test为包括自采数据在内的测试数据验证
```
```
demo.ipynb为演示所用文件，其内容完全被其他文件所包含，仅作为演示使用。
```
## 分类结果

对原始数据进行特征提取操作后，按照75%和25%的比例划分为训练集和验证集，并通过支持向量机进行训练与分类，在训练集和验证集上分别达到了98.2%和95.4%，同时，在15张测试数据上的准确率达到了100%。详细的训练及分类过程可见[演示文件](./demo.ipynb)。