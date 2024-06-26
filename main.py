import os
import imutils
import cv2 as cv
import numpy as np
import joblib
from scipy.cluster.vq import *
from train import data_process
from tqdm import tqdm
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 加载提取的特征向量和类标签
im_features, voc, idf, classes_names, stdSlr = joblib.load("features100noidf.pkl")
# im_features = im_features * idf

# 训练SVM
clf = svm.SVC(C=1, kernel='linear')
clf.fit(im_features, np.array(classes_names))

# 加载SIFT特征提取器
sift = cv.xfeatures2d.SIFT_create()

def predict_image(image_path):
    im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    im = cv.resize(im, (300, 300))
    # 显示图片
    # cv.imshow("im", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    kpts = sift.detect(im)
        # print('start sift')
    kpts, des = sift.compute(im, kpts)
    words, distance = vq(des, voc)
    im_features = np.zeros((1, 100), "float32")
    for w in words:
        im_features[0][w] += 1
    im_features = stdSlr.transform(im_features) # *idf
    return im_features
# 预测
image_paths, image_labels = data_process("test")
accuracy = 0
pred_list = []
# print(len(image_labels))
for image_path, image_label in tqdm(zip(image_paths, image_labels), total=len(image_paths)):
    im_features = predict_image(image_path)
    pred = clf.predict(im_features)
    # print("image: %s, classes : %s"%(image_path, pred))
    if pred == image_label:
        accuracy += 1
    pred_list.append(pred[0])
print("accuracy : ", accuracy / len(image_paths))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 假设pred_list和image_labels已经被定义：
# pred_list是你的模型预测的标签列表
# image_labels是正确的标签列表

# 生成混淆矩阵
cm = confusion_matrix(image_labels, pred_list)

# 使用seaborn来绘制格式化好的混淆矩阵
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# 为了使其更具可读性，你可以添加以下内容：
title = 'Confusion matrix'
# tick_marks，即轴标签上的标记位置
# 对于每个标记位置，你还可以包括类别名

plt.title(title, size = 15)
class_names = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']
# 添加轴标签（如果已知类别名称）
tick_marks = np.arange(len(class_names)) # class_names是你的类别名称列表
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# 显示混淆矩阵图像
plt.show()