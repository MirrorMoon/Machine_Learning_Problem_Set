import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np


def set_ax_gray(ax):
    # 设置坐标轴背景为灰色和网格样式
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)  # 设置背景的透明度
    ax.spines['right'].set_color('none')  # 隐藏右边的坐标轴线
    ax.spines['top'].set_color('none')    # 隐藏上边的坐标轴线
    ax.spines['bottom'].set_color('none') # 隐藏底边的坐标轴线
    ax.spines['left'].set_color('none')   # 隐藏左边的坐标轴线
    ax.grid(axis='y', linestyle='-.')     # 添加虚线风格的y轴网格线


# 读取数据文件，路径为本地存储路径
path = r'D:\Machine_Learning_Problem_Set\data\watermelon3_0a_Ch.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)  # 按空格分隔符读取数据文件

# 提取数据的特征和目标变量
X = data.iloc[:, [0]].values   # 读取第一列（密度）作为特征
y = data.iloc[:, 1].values     # 读取第二列（含糖率）作为目标值

# 初始化回归模型参数
gamma = 10  # RBF核函数的gamma参数
C = 1       # 惩罚系数C（防止过拟合）

# 创建绘图对象并设置坐标轴风格
ax = plt.subplot()  # 创建一个绘图子图
set_ax_gray(ax)     # 应用灰色背景和定制网格
ax.scatter(X, y, color='blue', label='data')  # 绘制原始数据点，颜色为蓝色

# 循环不同的gamma值以观察模型拟合的差异
for gamma in [1, 10, 100, 1000]:  # 测试不同的gamma值
    svr = svm.SVR(kernel='rbf', gamma=gamma, C=C)  # 使用RBF核的支持向量回归模型
    svr.fit(X, y)  # 训练模型

    # 预测模型输出并绘制回归曲线
    ax.plot(np.linspace(0.2, 0.8), svr.predict(np.linspace(0.2, 0.8).reshape(-1, 1)),
            label='gamma={}, C={}'.format(gamma, C))  # 在图中绘制不同gamma值下的回归曲线
ax.legend(loc='upper left')  # 将图例放置在左上角
ax.set_xlabel('密度')  # 设置x轴标签
ax.set_ylabel('含糖率')  # 设置y轴标签

plt.show()  # 显示绘制的图像
