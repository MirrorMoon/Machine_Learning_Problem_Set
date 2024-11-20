import numpy as np
import matplotlib.pyplot as plt

# 设置出图显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def decision_sdumps_MaxInfoGain(X, Y):
    # 基学习器---决策树桩，即高度为2的决策树，只有一个根节点及其对应的叶子节点

    # 以信息增益最大来选择划分属性和划分点
    m, n = X.shape  # 样本数和特征数
    results = []  # 存储各个特征下的最佳划分点,左分支取值，右分支取值，信息增益
    for i in range(n):  # 遍历各个候选特征
        x = X[:, i]  # 样本在该特征下的取值
        #unique后自动排序
        x_values = np.unique(x)  # 当前特征的所有取值
        #1. `x_values` 是当前特征的所有取值，并且已经通过 `np.unique(x)` 去重和排序。
        #2. `x_values[1:]` 表示从第二个元素到最后一个元素的子数组。
        #3. `x_values[:-1]` 表示从第一个元素到倒数第二个元素的子数组。
        #4. `(x_values[1:] + x_values[:-1]) / 2` 计算相邻元素的平均值，即每两个相邻特征值的中点。这些中点就是候选划分点。
        #式4.1
        GainTotal = -(sum(Y==1) / m * np.log2(sum(Y==1) / m) + sum(Y==-1) / m * np.log2(sum(Y==-1) / m))
        ts = (x_values[1:] + x_values[:-1]) / 2  # 候选划分点
        Gains = []  # 存储各个划分点下的信息增益
        #选出最优划分点
        for t in ts:
            Gain = 0
            Y_left = Y[x <= t]  # 左分支样本的标记
            Dl = len(Y_left)  # 左分支样本数
            p1 = sum(Y_left == 1) / Dl  # 左分支正样本比例
            p0 = sum(Y_left == -1) / Dl  # 左分支负样本比例
            #式子4.8的后半部分
            #因为ENT本来就有负号，因此负负得正直接加就好了
            Gain += -Dl / m * (np.log2(p1 ** p1) + np.log2(p0 ** p0))

            Y_right = Y[x > t]  # 右分支样本的标记
            Dr = len(Y_right)  # 右分支总样本数
            p1 = sum(Y_right == 1) / Dr  # 右分支正样本比例
            p0 = sum(Y_right == -1) / Dr  # 右分支负样本比例
            Gain += -Dr / m * (np.log2(p1 ** p1) + np.log2(p0 ** p0))
            # Gain += Dr / m * (np.log2(p1 ** p1) + np.log2(p0 ** p0))
            Gains.append(GainTotal-Gain)
            # Gains.append(Gain)
        best_t = ts[np.argmax(Gains)]  # 当前特征下的最佳划分点
        best_gain = max(Gains)  # 当前特征下的最佳信息增益
        #如果这个划分点下的正类样本>=负类，则sum(Y[x <= best_t])>=0,那么这个表达式为真，其值为true,*2-1为1，否则为加*2-1为-1。这样就实现了取多数类的类别
        left_value = (sum(Y[x <= best_t]) >= 0) * 2 - 1  # 左分支取值(多数类的类别)
        right_value = (sum(Y[x > best_t]) >= 0) * 2 - 1  # 右分支取值(多数类的类别)
        results.append([best_t, left_value, right_value, best_gain])

    results = np.array(results)
    #- `results`是一个二维数组，每一行包含一个特征的最佳划分点、左分支取值、右分支取值和信息增益。
    # `results[:, -1]`提取`results`数组中每一行的最后一个元素，即每个特征的最佳信息增益。
    # `np.argmax(results[:, -1])`返回`results`数组中信息增益最大的那个特征的索引。

    #最终，`df`表示具有最大信息增益的划分特征的索引。

    #返回行索引，也就是特征索引
    df = np.argmax(results[:, -1])  # df表示divide_feature，划分特征
    #将最佳特征索引及其对应的最佳划分点、左分支取值、右分支和并返回
    h = [df] + list(results[df, :3])  # 划分特征,划分点,左枝取值，右枝取值，增益率
    return h


def predict(H, X1, X2):
    # 预测结果
    # 仅X1和X2两个特征,X1和X2同维度
    pre = np.zeros(X1.shape)
    for h in H:
        df, t, lv, rv = h  # 划分特征,划分点,左枝取值，右枝取值
        X = X1 if df == 0 else X2
        pre += (X <= t) * lv + (X > t) * rv
    return np.sign(pre)


# >>>>>西瓜数据集3.0α
X = np.array([[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
              [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
              [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37],
              [0.593, 0.042], [0.719, 0.103]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
m = len(Y)
# >>>>>Bagging
#迭代次数
T = 20
H = []  # 存储各个决策树桩，
# 每行为四元素列表，分别表示划分特征,划分点,左枝取值，右枝取值
H_pre = np.zeros(m)  # 存储每次迭代后H对于训练集的预测结果
error = []  # 存储每次迭代后H的训练误差
for t in range(T):
    boot_strap_sampling = np.random.randint(0, m, m)  # 产生m个随机数，这m个随机数是样本的下标
    Xbs = X[boot_strap_sampling]  # 自助采样
    Ybs = Y[boot_strap_sampling]  # 自助采样
    h = decision_sdumps_MaxInfoGain(Xbs, Ybs)  # 训练基学习器
    H.append(h)  # 存入基学习器
    # 计算并存储训练误差
    df, t, lv, rv = h  # 基学习器参数
    #因为(X[:, df] <= t)表示的是最优划分特征的左分支，且这是一个bool数组，因此乘以其类别（1，-1）后，false的为0，即非最优特征左分支
    #而true，也就是最优划分特征左分支，负类就是-1，而正类就是1，乘上true也就是1就能得到其正确类别，两个向量相加就是最后的预测结果
    Y_pre_h = (X[:, df] <= t) * lv + (X[:, df] > t) * rv  # 基学习器预测结果
    #这里实际上就是投票法来确定最终的预测结果，比如第一次1，第二次-1，第三次1，最后的结果就是1
    H_pre += Y_pre_h  # 更新集成预测结果
    #这里一样的，>=0的就认为是正类，其值为true1，反之为false0，*2-1就是将true转换为1，false转换为-1，然后在和标签向量Y做判断，最后求和/m即得错误率
    error.append(sum(((H_pre >= 0) * 2 - 1) != Y) / m)  # 当前集成预测的训练误差
H = np.array(H)

# >>>>>绘制训练误差变化曲线
plt.title('训练误差的变化')
plt.plot(range(1, T + 1), error, 'o-', markersize=2)
plt.xlabel('基学习器个数')
plt.ylabel('错误率')
plt.show()
# >>>>>观察结果
# 获取第一个特征的最小值和最大值
x1min, x1max = X[:, 0].min(), X[:, 0].max()
# 获取第二个特征的最小值和最大值
x2min, x2max = X[:, 1].min(), X[:, 1].max()
# 生成第一个特征的等间距数值序列，范围比实际数据范围稍大
#分别向左右偏移20%的范围
x1 = np.linspace(x1min - (x1max - x1min) * 0.2, x1max + (x1max - x1min) * 0.2, 100)
# 生成第二个特征的等间距数值序列，范围比实际数据范围稍大
x2 = np.linspace(x2min - (x2max - x2min) * 0.2, x2max + (x2max - x2min) * 0.2, 100)
# 生成网格点矩阵，用于绘制决策边界
#将一维列表x1,x2分别扩展为矩阵，即将x1的每个分量都扩展成一个100*1的向量，最后拼在一起
X1, X2 = np.meshgrid(x1, x2)

for t in [3, 5, 11, 15, 20]:
    plt.title('前%d个基学习器' % t)
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    # plt.contourf(X1,X2,Ypre)
    # 画样本数据点
    #分别选取正负类的两个特征值作为坐标绘制散点
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='+', c='r', s=100, label='好瓜')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='_', c='k', s=100, label='坏瓜')
    plt.legend()
    # 画基学习器划分边界
    for i in range(t):
        feature, point = H[i, :2]
        if feature == 0:
            #如果是密度那么用含糖量填充，得到一条垂线
            plt.plot([point, point], [x2min, x2max], 'k', linewidth=1)
        else:
            #反之，如果是含糖量，那么用密度填充，得到一条水平线
            plt.plot([x1min, x1max], [point, point], 'k', linewidth=1)
    # 画基集成效果的划分边界

    Ypre = predict(H[:t], X1, X2)
    #等高线能找到函数所有取值相同的点，而预测值为0就是就是决策点（因为能把1和-1分开），因此利用等高线和预测值为0的点就能找到决策边界

    plt.contour(X1, X2, Ypre, colors='r', linewidths=5, levels=[0])
    plt.show()
