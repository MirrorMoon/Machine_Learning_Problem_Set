# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:03:47 2020

@author: Administrator
"""
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#==========读取Yale图片数据===========
#Yale人脸数据为15人，每人11张照片
#下载到的Yale文件存储规律是：
#    Yale文件夹下有名称分别为1~15的15个子文件夹，
#    每个子文件夹下有s1.bmp~s11.bmp的11张图片

rootpath='Yale'  #Yale根文件夹所在路径，我这里将Yale文件夹放在当前目录下，若其他位置，改成相应路径即可
X=[]             #存储图片数据
for person in range(15):
    for num in range(11):
        path=rootpath+'/'+str(person+1)+'/s'+str(num+1)+'.bmp'
        img=Image.open(path)
        X.append(np.array(img).reshape(-1))
X=np.array(X)

#==========观察这15人的图片============
#只显示第一张图片s1
for i in range(3):
    for j in range(5):
        plt.subplot(3,5,i*5+j+1)
        plt.imshow(X[(i*5+j)*11,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('%d'%(i*5+j+1))
plt.show()

#========PCA主成分分析(d'=20)==========
pca=PCA(n_components=20)
Z=pca.fit_transform(X)   #输入X的shape为m×d,与教材中相反
W=pca.components_        #特征向量W，shape为d'×d，与教材中相反

#====可视化观察特征向量所对应的图像======
for i in range(5):
    for j in range(4):
        plt.subplot(4,5,i*4+j+1)
        plt.imshow(W[i*4+j,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('w%d'%(i*4+j+1))
plt.show()

#========观察重构后的15人图片===========
#只显示第一张图片s1
X_re=pca.inverse_transform(Z)
for i in range(3):
    for j in range(5):
        plt.subplot(3,5,i*5+j+1)
        plt.imshow(X_re[(i*5+j)*11,:].reshape(100,100),cmap='gray')
        plt.axis('off')
        plt.title('%d'%(i*5+j+1))
plt.show()
