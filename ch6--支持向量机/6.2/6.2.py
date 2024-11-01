# 导入所需的库
from sklearn import svm  # 支持向量机(SVM)模型
import pandas as pd  # 用于数据处理
from matplotlib import pyplot as plt  # 数据可视化
import numpy as np  # 数值计算

# 定义一个函数，用于设置坐标轴的灰色背景和样式
def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")  # 设置背景色为灰色
    ax.patch.set_alpha(0.1)  # 设置背景透明度
    ax.spines['right'].set_color('none')  # 隐藏右侧坐标轴
    ax.spines['top'].set_color('none')  # 隐藏顶部坐标轴
    ax.spines['bottom'].set_color('none')  # 隐藏底部坐标轴
    ax.spines['left'].set_color('none')  # 隐藏左侧坐标轴
    ax.grid(axis='y', linestyle='-.')  # 设置 y 轴网格样式为虚线

# 定义绘制支持向量机分类结果的函数
def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1  # 获取标签为1的样本索引
    neg = y_ == -1  # 获取标签为-1的样本索引
    ax = plt.subplot()  # 创建一个绘图子图

    # 定义网格范围，用于绘制决策边界
    x_tmp = np.linspace(0, 1, 600)  # x轴的网格点
    y_tmp = np.linspace(0, 0.8, 600)  # y轴的网格点
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)  # 创建网格

    # 使用训练好的模型预测网格点的分类值
    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    # 绘制分类决策边界
    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})  # 添加标签显示决策边界

    set_ax_gray(ax)  # 设置灰色背景

    # 绘制不同类别的散点
    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')  # 标签为1的点
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')  # 标签为-1的点

    # 标记支持向量
    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o', facecolors='none', edgecolors='g', s=150,
               label='support_vectors')  # 支持向量的空心绿色圆圈

    ax.legend()  # 添加图例
    ax.set_title('{} kernel, C={}'.format(kernel, c))  # 设置标题显示核类型和惩罚系数
    plt.show()  # 显示图像

# 读取数据
path = r'D:\Machine_Learning_Problem_Set\data\watermelon3_0a_Ch.txt'
data = pd.read_table(path, delimiter=' ', dtype=float)  # 读取txt数据，空格分隔，数据类型为浮点数

# 提取特征和标签
X = data.iloc[:, [0, 1]].values  # 取出前两列作为特征
y = data.iloc[:, 2].values  # 第三列作为标签

y[y == 0] = -1  # 将标签0转为-1，适应SVM的输入格式

C = 100  # 惩罚系数，用于控制分类的松弛度

# 训练高斯核的SVM模型
clf_rbf = svm.SVC(C=C)  # 创建支持向量分类器，默认使用高斯核（rbf）
clf_rbf.fit(X, y.astype(int))  # 训练模型，标签转为整数类型
print('高斯核：')
print('预测值：', clf_rbf.predict(X))  # 输出预测值
print('真实值：', y.astype(int))  # 输出真实标签
print('支持向量：', clf_rbf.support_)  # 输出支持向量的索引

print('-' * 40)  # 分隔线

# 训练线性核的SVM模型
clf_linear = svm.SVC(C=C, kernel='linear')  # 创建支持向量分类器，使用线性核
clf_linear.fit(X, y.astype(int))  # 训练模型
print('线性核：')
print('预测值：', clf_linear.predict(X))  # 输出预测值
print('真实值：', y.astype(int))  # 输出真实标签
print('支持向量：', clf_linear.support_)  # 输出支持向量的索引

# 绘制高斯核的SVM分类结果
plt_support_(clf_rbf, X, y, 'rbf', C)

# 绘制线性核的SVM分类结果
plt_support_(clf_linear, X, y, 'linear', C)