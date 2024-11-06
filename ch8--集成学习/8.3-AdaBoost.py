import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 定义树的节点类
class Node(object):
    def __init__(self):
        # 存储决策树桩的特征索引
        self.feature_index = None
        # 存储决策树桩的分割点
        self.split_point = None
        # 存储当前节点的深度
        self.deep = None
        # 左子树节点
        self.left_tree = None
        # 右子树节点
        self.right_tree = None
        # 若节点为叶节点，则记录叶节点的分类结果
        self.leaf_class = None

# 计算基尼指数
def gini(y, D):
    '''
    计算样本集y下的加权基尼指数
    :param y: 数据样本标签
    :param D: 样本权重
    :return: 加权后的基尼指数
    '''
    unique_class = np.unique(y)  # 获取标签种类
    total_weight = np.sum(D)     # 总权重

    gini = 1
    #书p79 式4.5
    for c in unique_class:
        #
        gini -= (np.sum(D[y == c]) / total_weight) ** 2

    return gini

# 计算单一特征的基尼指数，找到最优分割点
def calcMinGiniIndex(a, y, D):
    '''
    计算特征a下样本集y的基尼指数
    :param a: 单一特征值
    :param y: 数据样本标签
    :param D: 样本权重
    :return: 最小基尼指数和分割点
    '''
    #排序是为了确定分割点，因为是连续型特征
    feature = np.sort(a)  # 对特征值排序
    #对应基尼值公式中的|D|
    total_weight = np.sum(D)  # 总权重

    # 生成所有可能的分割点
    split_points = [(feature[i] + feature[i + 1]) / 2 for i in range(feature.shape[0] - 1)]


    min_gini = float('inf')
    min_gini_point = None

    # 遍历分割点，计算基尼指数，找出最优分割点
    for i in split_points:
        yv1 = y[a <= i]  # 分割点左侧的标签
        yv2 = y[a > i]   # 分割点右侧的标签
        #本来是频率，但是由于采用的概率分布，所以这里的权重都用样本的分布来表示，将每个样本的权重加起来就是总权重
        Dv1 = D[a <= i]  # 左侧样本权重
        Dv2 = D[a > i]   # 右侧样本权重
        #这里的total_weight就是书p79式4.6的|D|
        gini_tmp = (np.sum(Dv1) * gini(yv1, Dv1) + np.sum(Dv2) * gini(yv2, Dv2)) / total_weight

        if gini_tmp < min_gini:
            min_gini = gini_tmp
            min_gini_point = i

    return min_gini, min_gini_point

# 选择基尼指数最小的特征和分割点
def chooseFeatureToSplit(X, y, D):
    '''
    选择分割特征
    :param X: 样本特征矩阵
    :param y: 样本标签
    :param D: 样本权重
    :return: 特征索引和分割点
    '''
    # 对每个特征计算基尼指数
    gini0, split_point0 = calcMinGiniIndex(X[:, 0], y, D)
    gini1, split_point1 = calcMinGiniIndex(X[:, 1], y, D)

    # 返回基尼指数更小的特征索引和分割点
    if gini0 > gini1:
        return 1, split_point1
    else:
        return 0, split_point0


def createSingleTree(X, y, D, deep=0):
    '''
    创建一个深度为2的决策树
    :param X: 训练集特征
    :param y: 训练集标签
    :param D: 样本权重
    :param deep: 当前树的深度
    :return: 树的根节点
    '''
    node = Node()
    node.deep = deep


    # 达到深度限制或样本数小于等于2时，将当前节点设置为叶节点
    if (deep == 2) | (X.shape[0] <= 2):
        pos_weight = np.sum(D[y == 1])  # 计算正类样本权重
        neg_weight = np.sum(D[y == -1])  # 计算负类样本权重
        node.leaf_class = 1 if pos_weight > neg_weight else -1
        return node

    # 找到当前分割的最佳特征及其分割点
    feature_index, split_point = chooseFeatureToSplit(X, y, D)
    node.feature_index = feature_index
    node.split_point = split_point

    # 递归构建左右子树
    #获取最优特征中小于等于分割点的样本以及大于分割点的样本下标
    left = X[:, feature_index] <= split_point
    right = X[:, feature_index] > split_point
    # 将左右子树对应的参数传入递归构建
    node.left_tree = createSingleTree(X[left, :], y[left], D[left], deep + 1)
    node.right_tree = createSingleTree(X[right, :], y[right], D[right], deep + 1)

    return node

# 基于单棵树预测单个样本
def predictSingle(tree, x):
    '''
    预测单个样本
    :param tree: 决策树
    :param x: 单个样本特征
    :return: 预测标签
    '''
    #当前节点是叶子节点
    if tree.leaf_class is not None:
        return tree.leaf_class
    #当前节点是分支节点（属性），根据与最优划分点的关系，进行左右子树的递归，从而找到样本的类别
    if x[tree.feature_index] > tree.split_point:
        return predictSingle(tree.right_tree, x)
    else:
        return predictSingle(tree.left_tree, x)

# 基于单棵树预测所有样本
def predictBase(tree, X):
    '''
    基于单棵树预测所有样本
    :param tree: 决策树
    :param X: 特征矩阵
    :return: 预测结果
    '''
    return np.array([predictSingle(tree, X[i, :]) for i in range(X.shape[0])])

# AdaBoost算法训练函数
def adaBoostTrain(X, y, tree_num=25):
    '''
    使用深度为2的决策树作为基学习器，训练AdaBoost
    :param X: 样本特征
    :param y: 样本标签
    :param tree_num: 基学习器数量
    :return: 基学习器集合及其权重
    '''
    D = np.ones(y.shape) / y.shape  # 初始化样本权重

    trees = []  # 存储基学习器
    a = []  # 存储学习器权重
    agg_est = np.zeros(y.shape)  # 累积预测值

    for _ in range(tree_num):
        #根据数据D和分布Dt训练一个基学习器
        tree = createSingleTree(X, y, D)  # 构建决策树
        #这里的预测结果是一个列表
        hx = predictBase(tree, X)  # 基学习器预测结果
        #hx!=y是一个布尔值列表，可以用来查找那些预测结果与实际结果不一样的样本的权重（或者说概率）
        #最后累计求和就是错误率，伪代码地上
        err_rate = np.sum(D[hx != y])  # 计算误差率
        if (err_rate > 0.5) | (err_rate == 0):  # 若误差率不满足要求，则停止训练
            print('err_rate:', err_rate)
            break
        #max(err_rate, 1e-16)是为了防止分母为0
        at = np.log((1 - err_rate) / max(err_rate, 1e-16)) / 2  # 计算权重
        #更新累积预测值
        agg_est += at * hx
        #集成
        trees.append(tree)
        a.append(at)



        # 更新样本权重
        #先默认填充1
        err_index = np.ones(y.shape)
        #将预测正确的样本权重设为-1
        err_index[hx == y] = -1
        #最后再用err_index乘以at便可得到第七行后面的分情况讨论的式子
        D = D * np.exp(err_index * at)
        #归一化，确保是一个分布
        D = D / np.sum(D)
        #这一段代码实际上就是书p176式子8.19
    return trees, a, agg_est

# 预测函数
def adaBoostPredict(X, trees, a):
    agg_est = np.zeros((X.shape[0],))

    for tree, am in zip(trees, a):
        agg_est += am * predictBase(tree, X)

    result = np.ones((X.shape[0],))
    result[agg_est < 0] = -1
    return result.astype(int)

# 可视化决策边界
def pltAdaBoostDecisionBound(X_, y_, trees, a):
    # 获取正类和负类样本的布尔索引
    pos = y_ == 1
    neg = y_ == -1

    # 生成用于绘制决策边界的网格点，也就是图中的黄线
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.2, 0.7, 600)
    #网格化，每一行都作为1列
    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    # 使用AdaBoost模型预测网格点的分类结果
    #ravel是把二维表格拉成一维，然后再用np.c_把两个一维表格合并成一个二维表格
    Z_ = adaBoostPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], trees, a).reshape(X_tmp.shape)

    # 绘制决策边界
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    # 绘制正类和负类样本的散点图
    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')

    # 显示图例
    plt.legend()

    # 显示图像
    plt.show()
# def pltAdaBoostDecisionBound(X_, y_, trees, a):
#     pos = y_ == 1
#     neg = y_ == -1
#     x_tmp = np.linspace(0, 1, 600)
#     y_tmp = np.linspace(-0.2, 0.7, 600)
#
#     X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
#     Z_ = adaBoostPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], trees, a).reshape(X_tmp.shape)
#     plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)
#     plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
#     plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
#     plt.legend()
#     plt.show()

# 主函数，读取数据并训练AdaBoost
if __name__ == "__main__":
    data_path = r'..\data\watermelon3_0a_Ch.txt'
    data = pd.read_table(data_path, delimiter=' ')
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    y[y == 0] = -1

    trees, a, agg_est = adaBoostTrain(X, y)
    pltAdaBoostDecisionBound(X, y, trees, a)
