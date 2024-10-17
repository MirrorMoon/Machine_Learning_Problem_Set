import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

# 定义数据集的文件路径
data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch3--线性模型\3.4\Transfusion.txt'

# 使用 np.loadtxt 加载数据集，按逗号分隔，并将数据转换为整数类型
data = np.loadtxt(data_path, delimiter=',').astype(int)

# 将数据分为特征 (X) 和标签 (y)
X = data[:, :4]  # 获取前四列作为特征矩阵 X
y = data[:, 4]   # 获取第五列作为标签 y

# 获取特征矩阵的行数 m（样本数）和列数 n（特征数）
m, n = X.shape

# 对特征进行标准化，使每个特征的均值为0，标准差为1
X = (X - X.mean(0)) / X.std(0)

# 创建一个包含样本索引的数组并随机打乱索引顺序
index = np.arange(m)
np.random.shuffle(index)

# 根据打乱后的索引重新排列特征矩阵和标签
X = X[index]
y = y[index]

# 使用 sklearn 中自带的 API 进行 k-10 交叉验证
lr = linear_model.LogisticRegression(C=2)  # 初始化逻辑回归模型，C 是正则化参数

# 进行 10 折交叉验证，返回每折的分数
score = cross_val_score(lr, X, y, cv=10)

# 打印 10 折交叉验证的平均分数
print(score.mean())

# LOO (Leave-One-Out) 交叉验证
loo = LeaveOneOut()  # 初始化留一法交叉验证

accuracy = 0  # 初始化准确率变量
for train, test in loo.split(X, y):  # 对数据进行留一法拆分
    lr_ = linear_model.LogisticRegression(C=2)  # 初始化新的逻辑回归模型
    X_train = X[train]  # 获取训练集特征
    X_test = X[test]    # 获取测试集特征
    y_train = y[train]  # 获取训练集标签
    y_test = y[test]    # 获取测试集标签
    lr_.fit(X_train, y_train)  # 训练逻辑回归模型

    # 累加每次测试的准确率
    accuracy += lr_.score(X_test, y_test)

# 计算并打印留一法交叉验证的平均准确率
print(accuracy / m)

# 自己写一个 k-10 交叉验证的实现
num_split = int(m / 10)  # 计算每折的样本数量
score_my = []  # 初始化自定义交叉验证的分数列表
for i in range(10):  # 遍历每一折
    lr_ = linear_model.LogisticRegression(C=2)  # 初始化新的逻辑回归模型
    test_index = range(i * num_split, (i + 1) * num_split)  # 确定测试集的索引范围
    X_test_ = X[test_index]  # 获取测试集特征
    y_test_ = y[test_index]   # 获取测试集标签

    # 获取训练集特征和标签，去掉测试集样本
    X_train_ = np.delete(X, test_index, axis=0)
    y_train_ = np.delete(y, test_index, axis=0)

    lr_.fit(X_train_, y_train_)  # 训练逻辑回归模型

    # 将当前折的分数添加到列表
    score_my.append(lr_.score(X_test_, y_test_))

# 打印自定义 k-10 交叉验证的平均分数
print(np.mean(score_my))

# LOO 的自定义实现
score_my_loo = []  # 初始化留一法的分数列表
for i in range(m):  # 遍历每一个样本
    lr_ = linear_model.LogisticRegression(C=2)  # 初始化新的逻辑回归模型
    X_test_ = X[i, :]  # 获取当前样本作为测试集特征
    y_test_ = y[i]     # 获取当前样本作为测试集标签

    # 获取训练集特征和标签，去掉当前样本
    X_train_ = np.delete(X, i, axis=0)
    y_train_ = np.delete(y, i, axis=0)

    lr_.fit(X_train_, y_train_)  # 训练逻辑回归模型

    # 预测当前测试样本的标签，并判断预测是否正确
    score_my_loo.append(int(lr_.predict(X_test_.reshape(1, -1)) == y_test_))

# 打印自定义留一法交叉验证的平均分数
print(np.mean(score_my_loo))

# 结果都是类似
