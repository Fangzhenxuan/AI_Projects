# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:40:20 2018

@author: 31723
"""

import numpy as np
from sklearn import svm,datasets
import matplotlib.pyplot as plt
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 设定训练样本的数量，m<64
m = 45
# 类别个数
w = 10

# 将一组64张的图像转换为64*32256的矩阵
def read_pgm(string):
    image_64 = np.zeros(192*168)
    
    # 读取路径为string下全部的pgm图片
    coll = io.ImageCollection(string)
    for i in range(len(coll)):
        a = coll[i]
        a = a.astype(float)
        a = a.reshape(1,192*168)
        image_64 = np.row_stack((image_64,a))
    image_64 = np.delete(image_64,0,axis=0)
    return image_64

# 给样本矩阵加标签
def add_label(matrix,n):
    a = np.full(64,1)
    for i in range(2,n+1):
        a = np.hstack((a,np.full(64,i)))
    matrix = np.hstack((matrix,a.reshape(64*w,1)))
    return matrix


# 对样本作L2归一化
def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t



# 创建样本矩阵,64个样本为第一类,有32556个特征
def read_image(w):
    string = []
    for i in range(1,w+1):
        if i<10: 
            string.append('CroppedYale/yaleB' + '0' + str(i) + '/*.pgm')
        else:
            string.append('CroppedYale/yaleB' + str(i) + '/*.pgm')
    sample = np.zeros(32256)
    for i in string:
        if i!='CroppedYale/yaleB14/*.pgm':
            sample = np.vstack((sample,read_pgm(i)))
    sample = np.delete(sample,0,axis=0)
    
    return sample

# 创建样本矩阵image_matrix
image_matrix = read_image(w)

# L2归一化
image_matrix = maxminnorm(image_matrix)

# 用PCA降维
pca = PCA(n_components=150)
image_matrix = pca.fit_transform(image_matrix)
# print(image_matrix)


# 给样本矩阵加标签
image_matrix = add_label(image_matrix,w)

# 将样本矩阵按行随机打乱
image_matrix = np.random.permutation(image_matrix)

# 将前2*m个样本作为训练集
train = image_matrix[0:w*m,:]

# 将后64*w-2*m个样本作为测试集
test = image_matrix[w*m:64*w,:]

# 进行 SVM模型训练
svc_rbf = svm.SVC(kernel='rbf',C=500,gamma=1e-5)        # 径向基核
svc_poly = svm.SVC(kernel='poly')                 # 多项式核
svc_linear = svm.SVC(kernel='linear')                   # 线性核
svc_rbf.fit(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel()) 
svc_poly.fit(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())
svc_linear.fit(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())

# 对测试集进行测试
result_rbf = svc_rbf.predict(test[:,:train.shape[1]-1])
result_poly = svc_poly.predict(test[:,:train.shape[1]-1])
result_linear = svc_linear.predict(test[:,:train.shape[1]-1])

# 对结果进行准确率计算
count1,count2,count3 = 0,0,0
for i in range(64*w-w*m):
    if test[i,train.shape[1]-1]==result_rbf[i]:
        count1 += 1
    if test[i,train.shape[1]-1]==result_poly[i]:
        count2 += 1
    if test[i,train.shape[1]-1]==result_linear[i]:
        count3 += 1

accuracy_rbf = count1 / (64*w-w*m)  
accuracy_poly = count2 / (64*w-w*m)
accuracy_linear = count3 / (64*w-w*m)
print('径向基（RBF）核函数的accuracy:',accuracy_rbf)
print('多项式（Poly）核函数的accuracy:',accuracy_poly)
#print('线性（Linear）核函数的accuracy:',accuracy_linear)


# 网格法找最优参数
tuned_parameters_rbf = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5,1e-6],
                     'C': [1, 10, 100,500, 1000]}]
tuned_parameters_poly = [{'kernel': ['poly'], 'C': [0.01,0.1,1, 10, 100]}]

# rbf
grid_rbf = GridSearchCV(SVC(),tuned_parameters_rbf, cv=5)
grid_rbf.fit(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())
print("The best parameters are %s with a score of %0.2f"
      % (grid_rbf.best_params_, grid_rbf.best_score_))
# poly
grid_poly = GridSearchCV(SVC(),tuned_parameters_poly, cv=5)
grid_poly.fit(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())
print("The best parameters are %s with a score of %0.2f"
      % (grid_poly.best_params_, grid_poly.best_score_))




'''
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



# test_PCA(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())
# plot_PCA(train[:,:train.shape[1]-1],train[:,train.shape[1]-1:].ravel())

'''









