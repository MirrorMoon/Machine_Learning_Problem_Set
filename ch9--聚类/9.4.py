import numpy as np
import matplotlib.pyplot as plt

# ========首先编写两个函数==============
# k_means：实现k-means聚类算法
# points：用于绘图的辅助函数，根据样本点的坐标计算外围凸多边形的顶点坐标

def k_means(X, k, u0=None, MaxIte=30):
    # k-means聚类算法
    # 输入：
    #    X:样本数据，m×n数组
    #    k:聚类簇数目
    #    u0:初始聚类中心
    #    MaxIte:最大迭代次数
    # 输出:
    #    u:最终的聚类中心
    #    C:各个样本所属簇号
    #    erro:按(9.24)式计算的平方方差结果
    #    step:迭代次数

    m, n = X.shape  # 样本数和特征数
    if u0 is None:  # 随机选取k个样本作为初始中心点
        u0 = X[np.random.permutation(m)[:k], :]
    u = u0.copy()
    step = 0
    while True:
        step += 1
        u_old = u.copy()  # 上一次迭代的中心点
        dist = np.zeros([m, k])  # 存储各个样本到中心点的距离
        for kk in range(k):  # 计算距离
            #这里直接用全体样本X-u[kk]，就能得到全体样本到第kk个中心的距离
            #axis=1是按行加和，这样就得到了全体样本没开方的距离向量，最后开方即可
            #最后填充到dist数组kk列中
            dist[:, kk] = np.sqrt(np.sum((X - u[kk]) ** 2, axis=1))
        #这个循环结束后，就得到了每个样本到每个中心的距离

        #dist[i,j]表示第i个样本到第j个中心的距离
        #np.argmin(dist, axis=1)计算每一行的最小值的索引号，即每个样本所属的簇类号,axis=1表示按行进行操作
        #最后的C是一个向量，每个分量代表每个样本所属的簇类号
        C = np.argmin(dist, axis=1)  # 以距离最小的中心点索引号最为簇类号
        for kk in range(k):  # 更新聚类中心
            #C==kk获取当前簇类号为kk的样本的索引号，取出属于当前簇的样本，并按列计算均值，这样就得到了新的中心点
            u[kk] = X[C == kk, :].mean(axis=0)
        if (u == u_old).all() or step > MaxIte:
            break  # 如果聚类中心无变化，或者超过最大
            # 迭代次数，则退出迭代
    # =====计算平方误差(9.24)
    erro = 0
    for kk in range(k):
        erro += ((X[C == kk] - u[kk]) ** 2).sum()
    return u, C, erro, step


def points(X, zoom=1.2):
    # 为了绘制出教材上那种凸聚类簇效果
    # 本函数用于计算凸多边形的各个顶点坐标
    # 输入:
    #     X:簇类样本点的坐标
    #     zoom:缩放因子(最外围样本点向外扩展系数)
    # 输出:
    #     ps:凸多边形的顶点坐标

    X = X[:, 0] + X[:, 1] * 1j  # 将坐标复数化
    u = np.mean(X)  # 聚类中心
    X = X - u  # 原点移至聚类中心
    # 寻找凸多边形的各个顶点坐标
    indexs = []  # 存储顶点坐标的索引号
    indexs.append(np.argmax(abs(X)))  # 首先将距离中心最远的点作为起始顶点
    while True:
        p = X[indexs[-1]]  # 当前最新确定的顶点
        X1 = 1E-5 + (X - p) / (-p)  # 以p点为原点，并且以u-p为x轴(角度为0)
        # 上式中加1E-5的小正数是因为p点自己减去自己的坐标有时候会出现
        # (-0+0j)的情况，angle(-0+0j)=-180°,但是希望结果为0
        indexs.append(np.argmax(np.angle(X1)))  # 新找到顶点
        if indexs[-1] == indexs[0]:  # 如果这些顶点首尾相连了，则停止
            break
    # 将复数坐标还原成直角坐标
    ps = np.c_[np.real(X)[indexs], np.imag(X)[indexs]]
    ps = ps * zoom + [np.real(u), np.imag(u)]
    return ps


# ================================================
#                  主程序
# ================================================

# ==============西瓜数据集4.0======================
D = np.array(
    [[0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
     [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
     [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
     [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
     [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
     [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])
m = D.shape[0]
# =============绘制样本数据点及其编号===============
plt.figure()
plt.scatter(D[:, 0], D[:, 1], marker='o', s=250, c='r', edgecolor='k')
for i in range(m):
    plt.text(D[i, 0], D[i, 1], str(i),
             horizontalalignment='center',
             verticalalignment='center')
plt.show()

# 选取三种不同的初始聚类中心点
centers = np.array([[5, 7, 17, 18],  # 相互靠近的点
                    [29, 15, 10, 25],  # 分散而外周的点
                    [27, 17, 12, 21]])  # 分散而中间的点

# ======运行k-means聚类，设置三组不同的k值、三组不同初始中心点=======
for i, k in enumerate([2, 3, 4]):  # 不同的k值
    for j in range(3):  # 3次不同初始值
        # =====k-means算法
        u0 = D[centers[j][:k], :]
        u, C, erro, step = k_means(D, k, u0)
        # =====画图======
        '''
        plt.subplot(3, 3, i * 3 + j + 1) 是一行使用 matplotlib.pyplot 库在图形中创建子图的 Python 代码。以下是详细解释：

        plt.subplot(3, 3, i * 3 + j + 1)：
        plt.subplot 是 matplotlib.pyplot 模块中的一个函数，用于在一个图形中创建子图。
        前两个参数 3 和 3 指定子图网格的行数和列数。在这种情况下，它创建一个 3x3 的网格。
        第三个参数 i * 3 + j + 1 指定子图在网格中的索引。子图按行优先顺序（从左到右，从上到下）编号，从 1 到 9（对于 3x3 网格）。
        
        表达式 i * 3 + j + 1 计算子图的位置：
        i 和 j 是遍历网格行和列的循环变量。
        i * 3 计算当前行的起始索引。
        j 添加当前行内的列偏移量。
        + 1 将索引调整为 1 基，因为 plt.subplot 需要 1 基索引。
        
        例如，如果 i = 1 和 j = 2，索引将是 1 * 3 + 2 + 1 = 6，这将子图放置在 3x3 网格的第 6 个位置。
        '''
        plt.subplot(3, 3, i * 3 + j + 1)
        plt.axis('equal')
        plt.title('k=%d,step=%d,erro=%.2f' % (k, step, erro))
        # 画样本点
        plt.scatter(D[:, 0], D[:, 1], c='k', s=1)
        # 画聚类中心
        #初始聚类中心
        plt.scatter(u0[:, 0], u0[:, 1], marker='o', c='g', s=50, edgecolors='b')
        #最终聚类中心
        plt.scatter(u[:, 0], u[:, 1], marker='+', c='r', s=50)  # ,c='',s=80,edgecolors='g')
        '''
            plt.annotate('', xy=u[kk], xytext=u0[kk], arrowprops=dict(arrowstyle='->'), ha='center')
            ```
            
            这行代码使用 `matplotlib` 库中的 `annotate` 函数在图中绘制箭头。以下是各个参数的详细解释：
            
            - `''`: 注释文本为空字符串，因此不会显示任何文本。
            - `xy=u[kk]`: 箭头的终点坐标，表示最终聚类中心的坐标。
            - `xytext=u0[kk]`: 箭头的起点坐标，表示初始聚类中心的坐标。
            - `arrowprops=dict(arrowstyle='->')`: 定义箭头的样式，这里使用了简单的箭头样式 `'->'`。
            - `ha='center'`: 水平对齐方式为居中。
            
            这行代码的作用是在图中绘制从初始聚类中心到最终聚类中心的箭头，以直观地展示聚类中心的移动过程。
        '''
        for kk in range(k):
            plt.annotate('', xy=u[kk], xytext=u0[kk], arrowprops=dict(arrowstyle='->'), ha='center')
        # 画聚类边界
        for kk in range(k):
            ps = points(D[C == kk])
            plt.plot(ps[:, 0], ps[:, 1], '--r', linewidth=1)
plt.tight_layout()

plt.show()