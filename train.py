from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

from os.path import join as pjoin
import matplotlib.pyplot as plt


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib



from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face


minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

print ("Creating Network and loading parameters")

print ("1 MTCNN")
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn (sess, None)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_image (presion_dir, f):
    img = cv2.imread(pjoin (presion_dir, f))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_and_align_data(image, image_size, margin, gpu_memory_fraction):

    # 读取图片 
    img = image
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组 （参数分别是输入图片 脸部最小尺寸 三个网络 阈值 factor不清楚）
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # bounding_boxes 返回的是人脸的坐标
    # 如果检测出图片中不存在人脸 则直接返回，return 0（表示不存在人脸，跳过此图）
    if len(bounding_boxes) < 1:
        return 0,0,0
    else:
        crop=[]
        det=bounding_boxes

        det[:,0]=np.maximum(det[:,0], 0)
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])

        # det[:,0]=np.maximum(det[:,0]-margin/2, 0)
        # det[:,1]=np.maximum(det[:,1]-margin/2, 0)
        # det[:,2]=np.minimum(det[:,2]+margin/2, img_size[1])
        # det[:,3]=np.minimum(det[:,3]+margin/2, img_size[0])

        det=det.astype(int)

        for i in range(len(bounding_boxes)):
            temp_crop=img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
            aligned=misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
        crop_image=np.stack(crop)
            
        return det,crop_image,1

def load_data (data_dir):
    data = {}
    pics_ctr = 0
    # 2.3.1 加载出一个文件夹中的子文件夹
    for guy in os.listdir (data_dir):
        presion_dir = pjoin (data_dir, guy)
        # 2.3.2 将每个子文件夹中的图像都加载出来
        curr_pics = [read_image(presion_dir, f) for f in os.listdir(presion_dir)]
        # 2.3.3 存储每一个人的文件夹中的所有图片
        data[guy] = curr_pics
    return data

print ("2 FaceNet")
model_dir = './20170512-110547'
with tf.Graph().as_default():
    with tf.Session() as sess:
        # 2.1 加载模型
        
        facenet.load_model(model_dir)
        
        # 2.2 调用facenet训练模块中的placder tensor
        images_placeholder = tf.get_default_graph().get_tensor_by_name ("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name ("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
        # 2.3 加载数据集
        data = load_data ("./train_dir/")
        
        """
            此时data这个字典中存放的是每一个名字，和每个名字的图片数据
            下面的目的是获取每一张图片通过FaceNet后的128维人脸特征
        """
        
        keys = []
        for key in data:
            keys.append (key)
            print ('folder:{}, image numbers:{}'.format(key, len(data[key])))
        # 2.4 使用mtcnn获得每张图片中的人脸数量和位置，并通过facenet得到embedding编码
        train_x = []
        train_y = []
        
        for i in range(len(keys)):
            for x in data[keys[i]]:
                # 2.4.1 使用mtcnn获取每张图片的人脸数量和位置
                _, images, flag = load_and_align_data(x, 160, 44, 1.0)
                if flag:
                    # 2.4.2 计算每张图片的人脸embedding编码
                    feed_dict = {images_placeholder : images, phase_train_placeholder : False}
                    emb = sess.run (embeddings, feed_dict = feed_dict)
                    for xx in range(len(emb)):
                        print(type(emb[xx,:]),emb[xx,:].shape)
                        train_x.append(emb[xx,:])       
                        train_y.append(i)
        print('Embeddings完成，样本数为：{}'.format(len(train_x)))
        
# 3 数据结构类型转换
train_x = np.array (train_x)
print (train_x.shape)
train_x = train_x.reshape (-1, 128)
train_y = np.array (train_y)
print (train_x.shape)
print (train_y.shape)

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.3)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier(n_neighbors = 1)  
    model.fit(train_x, train_y)  
    return model  

model = knn_classifier(X_train,y_train)  
predict = model.predict(X_test)  

accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 
  
    
#save model
joblib.dump(model, './models/knn_classifier.model')

model = joblib.load('./models/knn_classifier.model')
predict = model.predict(X_test) 
accuracy = metrics.accuracy_score(y_test, predict)  
print ('accuracy: %.2f%%' % (100 * accuracy)  ) 