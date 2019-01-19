# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:51:47 2018

@author: 31723
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sonar = pd.read_csv('sonar.all-data',header=None,sep=',')
sonar1 = sonar.iloc[0:208,0:60]
sonar2 = np.mat(sonar1) 
d = 5




# 顺序前进法SFS
def SFS(d):
    
    best = np.zeros(60)           # best是每次求出来最好的个体
    #第一个特征选择单独最优的特征
    fitness_1 = np.zeros(60)
    for i in range(60):
        people = np.zeros(60)     # 初始化个体
        people[i] = 1
        fitness_1[i] = Jd(people,1)
    a = fitness_1.argmax()
    best[a] = 1
    
    for i in range(2,d+1):
        fitness_test = np.zeros((61-i,2))
        k = 0
        for j in range(60):
            sample = best.copy()
            if sample[j] == 0:
                sample[j] = 1
                fitness_test[k,0] = j
                fitness_test[k,1] = Jd(sample,i)
                k += 1
        
        fitness_test = fitness_test[np.lexsort(fitness_test.T)]    # 按最后一列排序
        c = fitness_test[0,0].astype(int)
        best[c] = 1
        
        #终止迭代的条件
        
       # best_fitness_1 = Jd(best,d)
    
    return Jd(best,d)



# 顺序后退法
def SBS(d):
    best = np.ones(60)      # best是每次选出来最好的个体
    # 从每一个特征剔除一个最不好的特征
    for i in range(1,61-d):     # 60-d为剔除的变量个数,i为已剔除的变量个数
        fitness_test = np.zeros((61-i,2))  # 列数为还剩下的变量个数+1
        k = 0
        for j in range(60):
            sample = best.copy()
            if sample[j] == 1:
                sample[j] = 0
                fitness_test[k,0] = j
                fitness_test[k,1] = Jd(sample,60-i)
                k += 1
        
        fitness_test = fitness_test[np.lexsort(fitness_test.T)]    # 按最后一列排序
        c = fitness_test[60-i,0].astype(int)     # 取适应度最不好的剔除
        best[c] = 0
    
    
    return best,Jd(best,d)
    
    
    



#个体适应度函数 Jd(x)，x是d维特征向量(1*60维的行向量,1表示选择该特征)
def Jd(x,d1):
    #从特征向量x中提取出相应的特征
    Feature = np.zeros(d1)        #数组Feature用来存 x选择的是哪d个特征
    k = 0
    for i in range(60):
        if x[i] == 1:
            Feature[k] = i
            k+=1
    
    #将30个特征从sonar2数据集中取出重组成一个208*d的矩阵sonar3
    sonar3 = np.zeros((208,1))
    for i in range(d1):
        p = Feature[i]
        p = p.astype(int)
        q = sonar2[:,p]
        q = q.reshape(208,1)
        sonar3 = np.append(sonar3,q,axis=1)
    sonar3 = np.delete(sonar3,0,axis=1)
    
    #求类间离散度矩阵Sb
    sonar3_1 = sonar3[0:97,:]        #sonar数据集分为两类
    sonar3_2 = sonar3[97:208,:]
    m = np.mean(sonar3,axis=0)       #总体均值向量
    m1 = np.mean(sonar3_1,axis=0)    #第一类的均值向量
    m2 = np.mean(sonar3_2,axis=0)    #第二类的均值向量
    m = m.reshape(d1,1)               #将均值向量转换为列向量以便于计算
    m1 = m1.reshape(d1,1)
    m2 = m2.reshape(d1,1)
    Sb = ((m1 - m).dot((m1 - m).T)*(97/208) + (m2 - m).dot((m2 - m).T)*(111/208)) #除以类别个数
  #  Sb = ((m1 - m).dot((m1 - m).T) + (m2 - m).dot((m2 - m).T)) / 2 #除以类别个数
    #求类内离散度矩阵Sw
    S1 = np.zeros((d1,d1))
    S2 = np.zeros((d1,d1))
    for i in range(97):
        S1 += (sonar3_1[i].reshape(d1,1)-m1).dot((sonar3_1[i].reshape(d1,1)-m1).T)
    S1 = S1/97
    for i in range(111):
        S2 += (sonar3_2[i].reshape(d1,1)-m2).dot((sonar3_2[i].reshape(d1,1)-m2).T)
    S2 = S2/111
    
    Sw = (S1*(97/208) + S2*(111/208))
   # Sw = (S1 + S2) / 2
    #计算个体适应度函数 Jd(x)
    J1 = np.trace(Sb)
    J2 = np.trace(Sw)
    Jd = J1/J2
    
    return Jd
    


if __name__ == '__main__':
    
    '''
    d = 30
    #best_SFS,best_fitness = SFS()
    best_SBS,best_fitness = SBS(d)
    print("采用顺序后退法所选出的最佳染色体为：")
    choice = np.zeros(d)
    k = 0
    print(best_SBS)
    print("所选择的最优特征为：")
    for i in range (60):
        if best_SBS[i]==1:
            choice[k] = i+1
            k+=1
    print(choice)
    print("最优染色体的适应度值为：")
    print(best_fitness)
    '''
    
    
    best_d = np.zeros(60)
    for d in range(1,60):
        best_fitness_SFS = SFS(d)
        best_d[d-1] = best_fitness_SFS
        print(best_fitness_SFS)
    
   
    '''
    #画图
    x = np.arange(0,t,1)
    plt.xlabel('dimension')
    plt.ylabel('fitness')
    plt.ylim((0,0.1))            # y坐标的范围
    plt.plot(x,fitness_change,'r')
    '''
    
    





        














