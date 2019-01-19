# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 23:04:06 2018

数据集大小分别为 (7291, 256) (7291, 1) (2007, 256) (2007, 1)
@author: 31723
"""
import h5py
import pandas as pd
import numpy as np

# 读取 USPS数据集
def pre_handle():
    with h5py.File( 'usps.h5') as hf:
            train = hf.get('train')
            x_train = train.get('data')[:]
            y_train = train.get('target')[:]
            test = hf.get('test')
            x_test = test.get('data')[:]
            y_test = test.get('target')[:]

    train_data=pd.DataFrame(x_train)
    train_label=pd.DataFrame(y_train)
    test_data=pd.DataFrame(x_test)
    test_label=pd.DataFrame(y_test)
    return train_data,train_label,test_data,test_label


train_data,train_label,test_data,test_label = pre_handle()


train_data = np.mat(train_data)
train_label = np.mat(train_label)
test_data = np.mat(test_data)
test_label = np.mat(test_label)
train_label = train_label.astype(int)
test_label = test_label.astype(int)

def k_nn(train_data,train_label,test_data,test_label):
    accuracy = 0
    for i in range(2007):
        count = np.zeros(10)
        prediction = 0
        distance = np.zeros((7291,2))
      # test1 = test[:,0:60]
      # train1 = train[:,0:60]
        for t in range(7291):
            distance[t,1] = np.linalg.norm(test_data[i] - train_data[t])
            distance[t,0] = train_label[t]              # 储存标签和欧式距离
        order = distance[np.lexsort(distance.T)]    # 按最后一列排序
        
        for n in range(k):
            a = order[n,0]
            a = a.astype(int)
            count[a] += 1   
        prediction = count.argmax()                           # 取出现次数最多的为预测值
        if prediction == test_label[i]:
            accuracy += 1
    Accuracy = accuracy/2007
    print("USPS数据集的最近邻准确率为:",Accuracy)
    return Accuracy

Res = np.zeros(20)

for m in range(1):
    k = m+1
    Res[m] = k_nn(train_data,train_label,test_data,test_label)

'''
# 绘制 k与分类准确率的图像
import matplotlib.pyplot as plt

x = np.arange(1,21,1)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.ylim((0.5,1))            # y坐标的范围
#画图
plt.plot(x,Res,'r')
plt.savefig("k近邻_Usps.jpg",dpi=2000)

'''































