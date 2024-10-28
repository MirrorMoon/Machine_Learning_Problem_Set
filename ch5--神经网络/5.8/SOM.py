import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# 利用高斯距离法计算邻近点的权重
# X,Y 模板大小，c 中心点的位置，sigma 影响半径
def gaussion_neighborhood(X, Y, c, sigma):
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y))  # 创建网格坐标
    d = 2 * sigma * sigma  # 距离的标准化因子
    ax = np.exp(-np.power(xx - xx.T[c], 2) / d)  # 高斯函数计算x方向的距离衰减
    ay = np.exp(-np.power(yy - yy.T[c], 2) / d)  # 高斯函数计算y方向的距离衰减
    return (ax * ay).T  # 返回邻域矩阵

# 利用bubble距离法计算邻近点的权重
# X,Y 模板大小，c 中心点的位置，sigma 影响半径
def bubble_neighborhood(X, Y, c, sigma):
    neigx = np.arange(X)  # x轴邻域坐标
    neigY = np.arange(Y)  # y轴邻域坐标

    ax = np.logical_and(neigx > c[0] - sigma, neigx < c[0] + sigma)  # x方向邻域范围
    ay = np.logical_and(neigy > c[1] - sigma, neigy < c[1] + sigma)  # y方向邻域范围
    return np.outer(ax, ay) * 1.  # 返回邻域矩阵（0或1）

# 计算学习率随训练步骤的变化
def get_learning_rate(lr, t, max_steps):
    return lr / (1 + t / (max_steps / 2))  # 递减学习率

# 计算欧氏距离
def euclidean_distance(x, w):
    dis = np.expand_dims(x, axis=(0, 1)) - w  # 扩展维度并计算样本与所有权重的距离
    return np.linalg.norm(dis, axis=-1)  # 返回欧氏距离

# 对输入特征进行标准化 (x-mu)/std
def feature_normalization(data):
    mu = np.mean(data, axis=0, keepdims=True)  # 均值
    sigma = np.std(data, axis=0, keepdims=True)  # 标准差
    return (data - mu) / sigma  # 返回标准化数据

# 获取激活节点的位置（最小距离的权重节点）
def get_winner_index(x, w, dis_fun=euclidean_distance):
    dis = dis_fun(x, w)  # 计算输入样本与每个节点的距离
    index = np.where(dis == np.min(dis))  # 找到距离最小的位置
    return (index[0][0], index[1][0])  # 返回激活节点坐标

# 使用主成分分析（PCA）初始化权重
def weights_PCA(X, Y, data):
    N, D = np.shape(data)  # 获取数据集的大小
    weights = np.zeros([X, Y, D])  # 初始化权重矩阵

    pc_length, pc = np.linalg.eig(np.cov(np.transpose(data)))  # 计算数据协方差矩阵的特征向量
    pc_order = np.argsort(-pc_length)  # 按主成分的方差大小排序
    for i, c1 in enumerate(np.linspace(-1, 1, X)):  # 使用前两主成分生成权重
        for j, c2 in enumerate(np.linspace(-1, 1, Y)):
            weights[i, j] = c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]]
    return weights  # 返回初始化后的权重矩阵

# 计算量化误差（用于衡量SOM的性能）
def get_quantization_error(datas, weights):
    w_x, w_y = zip(*[get_winner_index(d, weights) for d in datas])  # 获取所有样本的激活节点
    error = datas - weights[w_x, w_y]  # 计算误差
    error = np.linalg.norm(error, axis=-1)  # 欧氏距离
    return np.mean(error)  # 返回平均量化误差

# 训练SOM网络
def train_SOM(X, Y, N_epoch, datas, init_lr=0.5, sigma=0.5, dis_fun=euclidean_distance, neighborhood_fun=gaussion_neighborhood, init_weight_fun=None, seed=20):
    N, D = np.shape(datas)  # 获取输入特征的维度
    N_steps = N_epoch * N  # 总训练步数

    rng = np.random.RandomState(seed)  # 固定随机种子
    if init_weight_fun is None:
        weights = rng.rand(X, Y, D) * 2 - 1  # 随机初始化权重
        weights /= np.linalg.norm(weights, axis=-1, keepdims=True)  # 正则化权重
    else:
        weights = init_weight_fun(X, Y, datas)  # 使用PCA初始化权重

    for n_epoch in range(N_epoch):
        print("Epoch %d" % (n_epoch + 1))
        index = rng.permutation(np.arange(N))  # 打乱样本顺序
        for n_step, _id in enumerate(index):
            x = datas[_id]  # 获取当前样本
            t = N * n_epoch + n_step  # 当前训练步数
            eta = get_learning_rate(init_lr, t, N_steps)  # 学习率

            winner = get_winner_index(x, weights, dis_fun)  # 激活节点
            new_sigma = get_learning_rate(sigma, t, N_steps)  # 邻域半径
            g = neighborhood_fun(X, Y, winner, new_sigma)  # 邻域权重
            g = g * eta  # 调整权重矩阵

            weights = weights + np.expand_dims(g, -1) * (x - weights)  # 更新权重

        print("quantization_error= %.4f" % (get_quantization_error(datas, weights)))  # 输出量化误差

    return weights  # 返回训练后的权重矩阵

# 计算U矩阵（显示相邻节点间的距离）
def get_U_Matrix(weights):
    X, Y, D = np.shape(weights)
    um = np.nan * np.zeros((X, Y, 8))  # 8邻域

    ii = [0, -1, -1, -1, 0, 1, 1, 1]
    jj = [-1, -1, 0, 1, 1, 1, 0, -1]

    for x in range(X):
        for y in range(Y):
            w_2 = weights[x, y]

            for k, (i, j) in enumerate(zip(ii, jj)):
                if (x + i >= 0 and x + i < X and y + j >= 0 and y + j < Y):
                    w_1 = weights[x + i, y + j]
                    um[x, y, k] = np.linalg.norm(w_1 - w_2)  # 计算相邻节点的距离

    um = np.nansum(um, axis=2)  # 计算所有方向上的距离和
    return um / um.max()  # 返回归一化的U矩阵

# 主函数
if __name__ == "__main__":
    columns = ['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel', 'asymmetry_coefficient', 'length_kernel_groove', 'target']
    data = pd.read_csv('seeds_dataset.txt', names=columns, sep='\t+', engine='python')  # 读取数据集
    labs = data['target'].values  # 标签
    label_names = {1: 'Kama', 2: 'Rosa', 3: 'Canadian'}
    datas = data[data.columns[:-1]].values  # 获取特征数据
    N, D = np.shape(datas)
    print(N, D)

    datas = feature_normalization(datas)  # 标准化数据

    weights = train_SOM(X=9, Y=9, N_epoch=4, datas=datas, sigma=1.5, init_weight_fun=weights_PCA)  # 训练SOM网络

    UM = get_U_Matrix(weights)  # 获取U矩阵

    plt.figure(figsize=(9, 9))
    plt.pcolor(UM.T, cmap='bone_r')  # 绘制距离图
    plt.colorbar()

    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']

    for i in range(N):
        x = datas[i]
        w = get_winner_index(x, weights)
        i_lab = labs[i] - 1

        plt.plot(w[0] + .5, w[1] + .5, markers[i_lab], markerfacecolor='None',
                 markeredgecolor=colors[i_lab], markersize=12, markeredgewidth=2)

    plt.show()  # 显示图形
