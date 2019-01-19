# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:44:13 2018

@author: 31723
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold

sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
data = np.mat(sonar1)


k = 2       # k为聚类的类别数
n = 208     # n为样本总个数
d = 60      # t为数据集的特征数

# k-means算法
def k_means():
    
    # 随机选k个初始聚类中心,聚类中心为每一类的均值向量
    m = np.zeros((k,d))
    for i in range(k):
        m[i] = data[np.random.randint(0,n)]
    
    # k_means聚类
    m_new = m.copy()
    t = 0
    while (1):       
        # 更新聚类中心
        m[0] = m_new[0]
        m[1] = m_new[1]
        w1 = np.zeros((1,d))
        w2 = np.zeros((1,d))
        # 将每一个样本按照欧式距离聚类
        for i in range(n):
            distance = np.zeros(k)      
            sample = data[i]
            for j in range(k):      # 将每一个样本与聚类中心比较
                distance[j] = np.linalg.norm(sample - m[j])
            category = distance.argmin()
            if category==0:
                w1 = np.row_stack((w1,sample))
            if category==1:
                w2 = np.row_stack((w2,sample))
        
        # 新的聚类中心
        w1 = np.delete(w1,0,axis=0)
        w2 = np.delete(w2,0,axis=0)
        m_new[0] = np.mean(w1,axis=0)
        m_new[1] = np.mean(w2,axis=0)


  
        # 聚类中心不再改变时，聚类停止
        if (m[0]==m_new[0]).all() and (m[1]==m_new[1]).all():
            break
    
        print(t)
        t+=1
        
        w = np.vstack((w1,w2))
        label1 = np.zeros((len(w1),1))
        label2 = np.ones((len(w2),1))
        label = np.vstack((label1,label2))
        label = np.ravel(label)
        test_PCA(w,label)
        plot_PCA(w,label)
    return w1,w2

def test_PCA(*data):
    X,Y=data
    pca=decomposition.PCA(n_components=None)
    pca.fit(X)
   # print("explained variance ratio:%s"%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X,Y=data
    pca=decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r=pca.transform(X)
 #   print(X_r)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0.33,0.33,0.33),)
    for label,color in zip(np.unique(Y),colors):
        position=Y==label
        ax.scatter(X_r[position,0],X_r[position,1],label="category=%d"%label,color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()

if __name__ == '__main__':
    w1,w2 = k_means()
    

    #print(w1)
    print("第一类的聚类样本数为：")
    print(w1.shape[0])
    #print(w2)
    print("第二类的聚类样本数为：")
    print(w2.shape[0])    
    



























