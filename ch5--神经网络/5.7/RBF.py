'''
这里使用均方根误差（RMS）作为损失函数的RBF（径向基函数）神经网络。
'''

import numpy as np
import matplotlib.pyplot as plt

# RBF网络的前向传播函数
def RBF_forward(X_, parameters_):
    #因为输入层到输出层使用的是非线性变换，所以只需要定义隐藏层到输出层的权重即可
    m, n = X_.shape  # m为样本数，n为特征维度
    beta = parameters_['beta']  # 径向基函数的扩展参数（影响高斯核函数的宽度）
    W = parameters_['W']  # 输出层的权重矩阵
    c = parameters_['c']  # 径向基函数的中心点
    b = parameters_['b']  # 偏置项

    t_ = c.shape[0]  # 隐藏层的节点数量，即径向基函数的数量
    p = np.zeros((m,
                  t_))  # 存储隐藏层的激活值，对应公式5.19，每一行对应一个样本在不同神经元上的激活值
    x_c = np.zeros((m, t_))  # 存储x - c_{i}，表示样本与中心点之间的距离，因为一个神经元一个中心点

    # 计算每个样本到各中心点的距离，并应用高斯核函数进行转换
    for i in range(t_):
        x_c[:, i] = np.linalg.norm(X_ - c[[i],], axis=1) ** 2  # 计算欧几里得距离的平方

        p[:, i] = np.exp(-beta[0, i] * x_c[:, i])  # 通过高斯核函数计算激活值

    # 计算输出层的值 a，线性组合隐藏层输出并加上偏置，因为对于rbf网络而言，输出层没有激活函数，直接线性组合得到结果
    a = np.dot(p, W.T) + b
    return a, p, x_c


# RBF网络的反向传播函数
def RBF_backward(a_, y_, x_c, p_, parameters_):
    m, n = a_.shape  # m为样本数，n为输出层维度（这里n=1，因为这是一个二分类问题）
    grad = {}  # 用来存储梯度
    beta = parameters_['beta']
    W = parameters_['W']
    #因为da并不直接参与梯度更新，所以da除以m，如果da也除以m，dw也除以m，那么就是除以m^2，显然是不正确的
    #均方误差
    da = (a_ - y_) # 损失函数对输出层的偏导数，a_为预测值，y_为真实标签

    dw = np.dot(da.T, p_) / m  # 计算权重矩阵W的梯度
    db = np.sum(da, axis=0, keepdims=True) / m  # 计算偏置b的梯度
    dp = np.dot(da, W)  # 计算损失函数对隐藏层激活值p的偏导数
    #dbeta是用来更新输入到隐藏层的（因为原来的线性组合丢入激活函数被rbf替换了）
    dbeta = np.sum(dp * p_ * (-x_c), axis=0, keepdims=True) / m  # 计算径向基函数的扩展参数beta的梯度

    # 确保计算出的梯度维度正确
    assert dbeta.shape == beta.shape
    assert dw.shape == W.shape

    # 将梯度存入字典
    grad['dw'] = dw
    grad['dbeta'] = dbeta
    grad['db'] = db

    return grad


# 计算损失函数
def compute_cost(y_hat_, y_):
    m = y_.shape[0]  # 样本数
    loss = np.sum((y_hat_ - y_) ** 2) / (2 * m)  # 均方误差（MSE）作为损失函数
    return np.squeeze(loss)  # 去掉不必要的维度，使输出是一个标量


# RBF模型训练函数
def RBF_model(X_, y_, learning_rate, num_epochs, t):
    '''

    :param X_: 输入数据（训练集）
    :param y_: 真实标签
    :param learning_rate: 学习率，用于更新参数
    :param num_epochs: 训练的迭代次数
    :param t: 隐藏层节点（径向基函数）的数量
    :return: 返回训练后的模型参数和损失值
    '''
    parameters = {}
    np.random.seed(16)  # 设置随机种子，确保结果可重复

    # 初始化参数
    #RBF 的输出仅仅依赖于输入向量与某个中心向量之间的距离，所以当RBF的中心点确定以后，这种映射关系也就确定了
    # 因为rbf网络输入层到隐层的映射直接通过径向基函数实现，所以不需要定义输入层到隐层的权重
    parameters['beta'] = np.random.randn(1, t)  # 初始化径向基函数的方差参数beta
    parameters['W'] = np.zeros((1, t))  # t个隐层神经元的连接权
    parameters['c'] = np.random.rand(t, 2)  # 初始化中心点c，8行两列，每一行都是对应一个隐神经元中心点
    parameters['b'] = np.zeros([1, 1])  # 初始化偏置b，输出层只有一个节点，所以阈值只有一个
    costs = []  # 用来存储每次迭代的损失值

    # 进行num_epochs次训练
    for i in range(num_epochs):
        a, p, x_c = RBF_forward(X_, parameters)  # 前向传播，计算预测值a
        cost = compute_cost(a, y_)  # 计算损失值
        costs.append(cost)  # 记录损失值
        grad = RBF_backward(a, y_, x_c, p, parameters)  # 反向传播，计算梯度

        # 更新参数
        parameters['beta'] -= learning_rate * grad['dbeta']
        parameters['W'] -= learning_rate * grad['dw']
        parameters['b'] -= learning_rate * grad['db']

    return parameters, costs


# 使用训练好的模型进行预测
def predict(X_, parameters_):
    a, p, x_c = RBF_forward(X_, parameters_)  # 前向传播，计算预测值
    return a




# 示例数据集，X为输入数据，y为输出标签
X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([[1], [1], [0], [0]])

# 训练RBF模型
parameters, costs = RBF_model(X, y, 0.003, 10000, 8)

# 绘制损失函数的变化曲线
plt.plot(costs)
plt.show()

# 使用训练好的模型进行预测
print(predict(X, parameters))

# 梯度检验
# parameters = {}
# parameters['beta'] = np.random.randn(1, 2)  # 初始化径向基的方差
# parameters['W'] = np.random.randn(1, 2)  # 初始化
# parameters['c'] = np.array([[0.1, 0.1], [0.8, 0.8]])
# parameters['b'] = np.zeros([1, 1])
# a, p, x_c = RBF_forward(X, parameters)
#
# cost = compute_cost(a, y)
# grad = RBF_backward(a, y, x_c, p, parameters)
#
#
# parameters['b'][0, 0] += 1e-6
#
# a1, p1, x_c1 = RBF_forward(X, parameters)
# cost1 = compute_cost(a1, y)
# print(grad['db'])
#
# print((cost1 - cost) / 1e-6)
