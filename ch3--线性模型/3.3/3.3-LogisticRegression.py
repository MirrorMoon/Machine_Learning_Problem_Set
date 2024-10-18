
# '''
# 与原书不同，原书中一个样本xi 为列向量，本代码中一个样本xi为行向量
# 尝试了两种优化方法，梯度下降和牛顿法。两者结果基本相同，不过有时因初始化的原因，
# 会导致牛顿法中海森矩阵为奇异矩阵，np.linalg.inv(hess)会报错。以后有机会再写拟牛顿法吧。
# '''
# 导入需要的库
import numpy as np  # 用于矩阵运算
import pandas as pd  # 用于数据处理
from matplotlib import pyplot as plt  # 用于绘图
from sklearn import linear_model  # 用于调用sklearn的逻辑回归模型

# 定义sigmoid函数，用于将输入映射到0到1之间
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))  # sigmoid函数公式
    return s

# 定义成本函数J，计算给定参数beta下的损失值
def J_cost(X, y, beta):
    '''
    :param X: 样本数据数组，形状为(n_samples, n_features)
    :param y: 标签数组，形状为(n_samples,)
    :param beta: 参数向量，形状为(n_features + 1, )
    :return: 返回公式3.27的计算结果
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]  # 增加一列偏置项1
    beta = beta.reshape(-1, 1)  # 确保beta为列向量
    y = y.reshape(-1, 1)  # 确保y为列向量

    # 计算损失值：-y*X_hat*beta + log(1+exp(X_hat*beta))
    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

    return Lbeta.sum()  # 返回损失值的总和

# 计算损失函数的梯度（第一导数）
def gradient(X, y, beta):
    '''
    计算J相对于beta的梯度（公式3.27的第一导数），即公式3.30
    :param X: 样本数据数组，形状为(n_samples, n_features)
    :param y: 标签数组，形状为(n_samples,)
    :param beta: 参数向量，形状为(n_features + 1, )
    :return: 梯度向量
    '''
    #X.shape[0]作用时获取X的行数
    #np.ones的作用是生成一个行数为17，列数为1且元素都为1的数组/向量
    #np.c_的作用是将两个数组拼接为一个新的数组
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]  # 增加一列偏置项1
    #reshape重新调整形状，第一个参数-1自动获取函数，第二个参数1表示列数为1
    beta = beta.reshape(-1, 1)  # 确保beta为列向量
    y = y.reshape(-1, 1)  # 确保y为列向量
    #利用sigmoid函数计算预测值，由书不难得beta=w;b，因此对x_hat与beta做点击实际上得到的是wx+b
    #而后将其代入至sigmoid函数得到预测值
    p1 = sigmoid(np.dot(X_hat, beta))  # 计算预测值p1


    #因为我们的代价函数是极大似然估计函数，因此根据西瓜书可以得到其梯度为
    gra = (-X_hat * (y - p1)).sum(0)  # 计算梯度

    return gra.reshape(-1, 1)  # 返回列向量形式的梯度

# 计算海森矩阵（损失函数的二阶导数）
def hessian(X, y, beta):
    '''
    计算J相对于beta的二阶导数（即海森矩阵），公式3.31
    :param X: 样本数据数组，形状为(n_samples, n_features)
    :param y: 标签数组，形状为(n_samples,)
    :param beta: 参数向量，形状为(n_features + 1, )
    :return: 海森矩阵
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]  # 增加一列偏置项1
    beta = beta.reshape(-1, 1)  # 确保beta为列向量
    y = y.reshape(-1, 1)  # 确保y为列向量

    p1 = sigmoid(np.dot(X_hat, beta))  # 计算预测值p1

    m, n = X.shape  # 获取样本数量和特征数
    P = np.eye(m) * p1 * (1 - p1)  # 生成对角矩阵P，包含p1*(1-p1)

    assert P.shape[0] == P.shape[1]  # 确保P是方阵
    return np.dot(np.dot(X_hat.T, P), X_hat)  # 返回海森矩阵

# 使用梯度下降法更新参数
def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    '''
    使用梯度下降法更新参数
    :param beta: 参数向量
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印损失
    :return: 更新后的参数
    '''
    for i in range(num_iterations):
        #显然beta就是梯度下降中的迭代点，也就是w和b
        grad = gradient(X, y, beta)  # 计算梯度
        #梯度下降的迭代公式
        beta = beta - learning_rate * grad  # 更新参数

        if (i % 10 == 0) & print_cost:  # 每10次迭代打印损失值
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta  # 返回更新后的参数

# 使用牛顿法更新参数
def update_parameters_newton(X, y, beta, num_iterations, print_cost):
    '''
    使用牛顿法更新参数
    :param beta: 参数向量
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印损失
    :return: 更新后的参数
    '''
    for i in range(num_iterations):

        grad = gradient(X, y, beta)  # 计算梯度
        hess = hessian(X, y, beta)  # 计算海森矩阵
        beta = beta - np.dot(np.linalg.inv(hess), grad)  # 使用牛顿法更新参数

        if (i % 10 == 0) & print_cost:  # 每10次迭代打印损失值
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta  # 返回更新后的参数

# 初始化beta参数
def initialize_beta(n):
    beta = np.random.randn(n + 1, 1) * 0.5 + 1  # 初始化beta，随机生成n+1维的向量
    return beta

# 逻辑回归模型的主函数
def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    '''
    :param X: 输入数据
    :param y: 标签
    :param num_iterations: 迭代次数
    :param learning_rate: 学习率
    :param print_cost: 是否打印损失
    :param method: 使用的优化方法（梯度下降或牛顿法）
    :return: 优化后的参数
    '''
    m, n = X.shape  # 获取样本数量和特征数
    beta = initialize_beta(n)  # 初始化beta

    if method == 'gradDesc':  # 如果选择梯度下降
        return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
    elif method == 'Newton':  # 如果选择牛顿法
        return update_parameters_newton(X, y, beta, num_iterations, print_cost)
    else:  # 如果输入未知方法，抛出错误
        raise ValueError('Unknown solver %s' % method)

# 定义预测函数
def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]  # 增加一列偏置项1
    p1 = sigmoid(np.dot(X_hat, beta))  # 计算预测值

    p1[p1 >= 0.5] = 1  # 如果预测值>=0.5，则判为1
    p1[p1 < 0.5] = 0  # 如果预测值<0.5，则判为0

    return p1  # 返回预测结果

# 主程序
if __name__ == '__main__':
    data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch3--线性模型\3.3\watermelon3_0_Ch.csv'  # 数据路径
    data = pd.read_csv(data_path).values  # 读取CSV文件数据

    #data[:,9]提取所有行的第九列
    is_good = data[:, 9] == '是'  # 筛选出标签为“是”的样本
    is_bad = data[:, 9] == '否'  # 筛选出标签为“否”的样本
    #提取密度和含糖量
    X = data[:, 7:9].astype(float)  # 提取特征列
    y = data[:, 9]  # 提取标签列

    y[y == '是'] = 1  # 将“是”转为1
    y[y == '否'] = 0  # 将“否”转为0
    y = y.astype(int)  # 转为整数类型

    #通过布尔索引返回第七、八列中对应is_good或is_bad为true的数据
    plt.scatter(data[:, 7][is_good], data[:, 8][is_good], c='k', marker='o')  # 绘制标签为“是”的样本点
    plt.scatter(data[:, 7][is_bad], data[:, 8][is_bad], c='r', marker='x')  # 绘制标签为“否”的样本点

    plt.xlabel('密度')  # x轴标签
    plt.ylabel('含糖量')  # y轴标签

    # 可视化模型结果

    beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)  # 训练逻辑回归模型
    w1, w2, intercept = beta  # 得到训练后得w和b
    x1 = np.linspace(0, 1)  # 创建线性空间
    #即二元线性函数w1x1+w2x2+b中的x2
    y1 = -(w1 * x1 + intercept) / w2  # 计算决策边界

    ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')  # 绘制自实现模型的决策边界

    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 使用sklearn的逻辑回归模型
    lr.fit(X, y)  # 训练sklearn逻辑回归模型

    lr_beta = np.c_[lr.coef_, lr.intercept_]  # 获取sklearn模型的参数
    print(J_cost(X, y, lr_beta))  # 打印sklearn模型的损失值

    # 可视化sklearn模型结果
    w1_sk, w2_sk = lr.coef_[0, :]  # 提取sklearn模型的系数

    x2 = np.linspace(0, 1)  # 创建线性空间
    y2 = -(w1_sk * x2 + lr.intercept_) / w2  # 计算sklearn模型的决策边界

    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')  # 绘制sklearn模型的决策边界
    # 解决无法正常显示标签
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.legend(loc='upper right')  # 添加图例
    plt.show()  # 显示图像

