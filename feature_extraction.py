import cv2
import numpy as np
import os
from sklearn import svm
import joblib
import time
from scipy.cluster.vq import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def data_process(path):
    label_names = os.listdir(path)
    image_paths = []
    image_labels = []
    class_id = 0
    for label in label_names:
        dir = os.path.join(path, label)
        class_path = [os.path.join(dir, img_path) for img_path in os.listdir(dir)] # 获取当前类别下的所有图片路径
        image_paths += class_path # 获取当前类别下的所有图片路径
        image_labels += [class_id] * len(class_path) # 生成当前类别的标签
        # print(len(class_path))
        class_id += 1
    return image_paths, image_labels


def sift_feature(image_paths):
    # 创建SIFT特征提取器
    sift = cv2.xfeatures2d.SIFT_create()
    # 特征提取与描述子生成
    des_list = []
    for image_path in tqdm(image_paths, desc="SIFT feature extracting..."):
        im = cv2.imread(image_path)
        im = cv2.resize(im, (300, 300))
        kpts = sift.detect(im)
        kpts, des = sift.compute(im, kpts)
        des_list.append((image_path, des))
        # print("image file path : ", image_path)
    # 描述子向量
    descriptors = des_list[0][1]
    for image_path, descriptor in tqdm(des_list[1:], desc="SIFT feature stacking..."):
        descriptors = np.vstack((descriptors, descriptor)) # 垂直堆叠descriptors
    return des_list, descriptors



def bof_feature(des_list, descriptors, k=100):
    print("kmeans...")
    time_start = time.time()
    voc, _ = kmeans(descriptors, k, 1)
    time_end = time.time()
    print("time cost", time_end - time_start, "s")
    print("kmeans done...")
    # 生成特征直方图
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in tqdm(range(len(image_paths)), desc="BOF feature extracting..."):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # 实现动词词频与出现频率统计
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
    return im_features, voc, idf

if __name__ == "__main__":
    train_path = "train"
    image_paths, image_labels = data_process(train_path)
    # des_list, descriptors = sift_feature(image_paths)
    # joblib.dump((des_list, descriptors), "des_list.pkl", compress=3)

    des_list, descriptors = joblib.load("des_list.pkl")
    im_features, voc, idf = bof_feature(des_list, descriptors)
    im_features = im_features * idf
    # 尺度化
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    # 保存特征
    print("save features...")
    joblib.dump((im_features, voc, idf, image_labels, stdSlr), "features100noidf.pkl", compress=3)

    # # Train the Linear SVM
    # clf = svm.SVC(C=1000.0, kernel='linear', gamma='auto')
    # clf.fit(im_features, np.array(image_labels))

    # # Save the SVM
    # print("training and save model...")
    # joblib.dump((clf, image_labels, stdSlr, 100, voc), "bof.pkl", compress=3)