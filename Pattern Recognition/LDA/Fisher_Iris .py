# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:12:41 2018

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
    if c == 0:                        # 第一种情况
        for i in range(0,49):
            S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
        for i in range(0,50):
            S2 += (X2[i].reshape(n,1)-m2).dot((X2[i].reshape(n,1)-m2).T)
    if c == 1:
        for i in range(0,50):
            S1 += (X1[i].reshape(n,1)-m1).dot((X1[i].reshape(n,1)-m1).T)
        for i in range(0,49):
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
iris = pd.read_csv('iris.data',header=None,sep=',')
iris1 = iris.iloc[0:150,0:4]
iris2 = np.mat(iris1)

Accuracy = 0
accuracy_ = np.zeros(10)
   

P1 = iris2[0:50,0:4]
P2 = iris2[50:100,0:4]
P3 = iris2[100:150,0:4]
    
G121 = np.ones(50)
G122 = np.ones(50)
G131 = np.zeros(50)
G132 = np.zeros(50)
G231 = np.zeros(50)
G232 = np.zeros(50)
   
# 留一法验证准确性
# 第一类和第二类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        test = P1[i]
        test = test.reshape(4,1)
        train = np.delete(P1,i,axis=0)       # 训练样本是一个列数为t的矩阵
        W,W0 = Fisher(train,P2,4,0)
        if (Classify(test,W,W0)) >= 0:
            count += 1
            G121[i] = Classify(test,W,W0)
    else:
        test = P2[i-50]
        test = test.reshape(4,1)
        train = np.delete(P2,i-50,axis=0)
        W,W0 = Fisher(P1,train,4,1)
        if (Classify(test,W,W0)) < 0:
            count += 1
            G122[i-50] = Classify(test,W,W0)
Accuracy12 = count/100
print("第一类和二类的分类准确率为:%.3f"%(Accuracy12))

# 第一类和第三类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        test = P1[i]
        test = test.reshape(4,1)
        train = np.delete(P1,i,axis=0)       # 训练样本是一个列数为t的矩阵
        W,W0 = Fisher(train,P3,4,0)
        if (Classify(test,W,W0)) >= 0:
            count += 1
            G131[i] = Classify(test,W,W0)
    else:
        test = P3[i-50]
        test = test.reshape(4,1)
        train = np.delete(P3,i-50,axis=0)
        W,W0 = Fisher(P1,train,4,1)
        if (Classify(test,W,W0)) < 0:
            count += 1
            G132[i-50] = Classify(test,W,W0)

Accuracy13 = count/100
print("第一类和三类的分类准确率为:%.3f"%(Accuracy13))

# 第二类和第三类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        test = P2[i]
        test = test.reshape(4,1)
        train = np.delete(P2,i,axis=0)       # 训练样本是一个列数为t的矩阵
        W,W0 = Fisher(train,P3,4,0)
        if (Classify(test,W,W0)) >= 0:
            count += 1
            G231[i] = Classify(test,W,W0)
    else:
        test = P3[i-50]
        test = test.reshape(4,1)
        train = np.delete(P3,i-50,axis=0)
        W,W0 = Fisher(P2,train,4,1)
        if (Classify(test,W,W0)) < 0:
            count += 1
            G232[i-50] = Classify(test,W,W0)

Accuracy23 = count/100
print("第二类和三类的分类准确率为:%.3f"%(Accuracy23))

# 画相关的图
import matplotlib.pyplot as plt

y1 = np.zeros(50)
y2 = np.zeros(50)
plt.figure(1)
plt.ylim((-0.5,0.5))            # y坐标的范围
#画散点图
plt.scatter(G121, y1,c='red', alpha=1, marker='.')
plt.scatter(G122, y2,c='k', alpha=1, marker='.')
plt.xlabel('Class:1-2')
plt.savefig('iris 1-2.png',dpi=2000)


plt.figure(2)
plt.ylim((-0.5,0.5))            # y坐标的范围
#画散点图
plt.scatter(G131, y1,c='red', alpha=1, marker='.')
plt.scatter(G132, y2,c='k', alpha=1, marker='.')
plt.xlabel('Class:1-3')
plt.savefig('iris 1-3.png',dpi=2000)


plt.figure(3)
plt.ylim((-0.5,0.5))            # y坐标的范围
#画散点图
plt.scatter(G231, y1,c='red', alpha=1, marker='.')
plt.scatter(G232, y2,c='k', alpha=1, marker='.')
plt.xlabel('Class:2-3')
plt.savefig('iris 2-3.png',dpi=2000)










