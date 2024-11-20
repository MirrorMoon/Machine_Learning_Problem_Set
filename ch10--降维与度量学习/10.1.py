# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:04:11 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
#设置绘图时显示中文
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def kNN(X,Y,Xpre,k,p=2):
    # k近邻算法
    # X：样本数据
    # Y：样本标记
    # Xpre：待预测样本
    # k：k近邻的k值
    # p:计算距离所采用的闵可夫斯基距离的p值
    mt,n=X.shape             #训练样本数和特征数
    Xpre=np.array(Xpre).reshape(-1,n)
    mp=Xpre.shape[0]         #预测样本数
    dist=np.zeros([mp,mt])   #存储预测样本和训练样本之间的距离
    for i in range(mt):
        #计算预测样本与训练样本的闵可夫斯基距离
        dist[:,i]=(((abs(Xpre-X[i]))**p).sum(axis=1))**(1/p)
    neighbor=np.argsort(dist,axis=1)   #训练样本按距离远近排序的索引号
    neighbor=neighbor[:,:k]            #只取前k个作为最近邻
    #获取k近邻类别
    Ypre=Y[neighbor]
    return (Ypre.sum(axis=1)>=0)*2-1   #西瓜3.0α仅两类，故可如此计算

# 西瓜3.0α 样本数据
X=np.array([[0.697,0.46],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
   [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
   [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.36,0.37],
   [0.593,0.042],[0.719,0.103]])
Y=np.array([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# 执行kNN算法
# 尝试 k=1,3,5,p=1,2,50的不同情况
ks=[1,3,5]
ps=[1,2,50]  #p=1为曼哈顿距离，p=2为欧式距离，p=50(→∞)为切比雪夫距离
for i,k in enumerate(ks):
    for j,p in enumerate(ps):
        # kNN算法预测结果
        x0=np.linspace(min(X[:,0]),max(X[:,0]),60)
        x1=np.linspace(min(X[:,1]),max(X[:,1]),60)
        X0,X1=np.meshgrid(x0,x1)
        #获取坐标对数组
        Xpre=np.c_[X0.reshape(-1,1),X1.reshape(-1,1)]
        Ypre=kNN(X,Y,Xpre,k,p).reshape(X0.shape)
        # 画图
        plt.subplot(len(ks),len(ps),i*len(ps)+j+1)
        #plt.axis('equal')
        plt.title('k=%d,p=%d'%(k,p))
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        # 画样本点
        plt.scatter(X[Y==1,0],X[Y==1,1],marker='+',s=30,label='好瓜')
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker='_',s=30,label='坏瓜')
        # 画决策树边界 (直接根据教材上图4.10和4.11确定边界曲线坐标)
        clf = tree.DecisionTreeClassifier().fit(X,Y)
        clfPre = clf.predict(Xpre).reshape(X0.shape)
        plt.contour(X0, X1, clfPre, colors='k', linewidths=1, levels=[0])
        # plt.plot([0.381,0.381,0.56,0.56,max(X[:,0])],
        #           [max(X[:,1]),0.126,0.126,0.205,0.205],'k',label='决策树边界')
        # 画kNN边界
        plt.contour(X0,X1,Ypre,1,colors='r',s=2)
plt.show()
