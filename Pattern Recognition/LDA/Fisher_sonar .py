# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:16:43 2018

@author: 31723
"""

import pandas as pd
import numpy as np


def Fisher(X1,X2,n,c):
    
    # 计算三类样本的类均值向量
    m1=(np.mean(X1,axis = 0))
    m2=(np.mean(X2,axis = 0))
    m1 = m1.reshape(n,1)   # 将行向量转换为列向量以便于计算
    m2 = m2.reshape(n,1)

    #计算类内离散度矩阵
    S1 = np.zeros((n,n))              # m1 = within_class_scatter_matrix1
    S2 = np.zeros((n,n))              # m2 = within_class_scatter_matrix2
    if c == 0:                          # 第一种情况
        for i in range(0,96):
            S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
        for i in range(0,111):
            S2 += (X2[i].reshape(n,1)-m2).dot((X2[i].reshape(n,1)-m2).T)
    if c == 1:
        for i in range(0,97):
            S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
        for i in range(0,110):
            S2 += (X2[i].reshape(n,1)-m2).dot((X2[i].reshape(n,1)-m2).T)
    #计算总类内离散度矩阵S_w
    S_w = S1 + S2

    #计算最优投影方向 W
    W = np.linalg.inv(S_w).dot(m1 - m2)
    #在投影后的一维空间求两类的均值
    m_1 = (W.T).dot(m1)
    m_2 = (W.T).dot(m2)
    
    #计算分类阈值 W0(为一个列向量)
    W0 = -0.5*(m_1 + m_2)
    
    return W,W0

def Classify(X,W,W0):
    y = (W.T).dot(X) + W0
    return y


#导入sonar.all-data数据集
sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1)

Accuracy = np.zeros(60)
accuracy_ = np.zeros(10)

for n in range(1,61):               # n是当前的维数
    for t in range(10):             # 每一维都求十次平均值
        sonar_random = (np.random.permutation(sonar2.T)).T    # 对原sonar数据进行每列打乱   

        P1 = sonar_random[0:97,0:n]
        P2 = sonar_random[97:208,0:n]
        
        count = 0
        #留一法验证准确性
        for i in range(208):
            if i <= 96:
                test = P1[i]
                test = test.reshape(n,1)
                train = np.delete(P1,i,axis=0)       # 训练样本是一个列数为t的矩阵
                W,W0 = Fisher(train,P2,n,0)
                if (Classify(test,W,W0)) >= 0:
                    count += 1
            else:
                test = P2[i-97]
                test = test.reshape(n,1)
                train = np.delete(P2,i-97,axis=0)
                W,W0 = Fisher(P1,train,n,1)
                if (Classify(test,W,W0)) < 0:
                    count += 1
        accuracy_[t] = count/208
    for k in range(10):
        Accuracy[n-1] += accuracy_[k]
    Accuracy[n-1] = Accuracy[n-1]/10
    print("当数据为%d维时，Accuracy:%.3f"%(n,Accuracy[n-1]))


# 画相关的图
import matplotlib.pyplot as plt

x = np.arange(1,61,1)
plt.xlabel('dimension')
plt.ylabel('Accuracy')
plt.ylim((0.5,0.8))            # y坐标的范围
#画图
plt.plot(x,Accuracy,'b')

























