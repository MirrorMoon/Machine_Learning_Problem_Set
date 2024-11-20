
import numpy as np
import matplotlib.pyplot as plt


# ==============首先编写几个函数，主程序见后==============
def relationship(net):
    # 计算网络中的每个结点的父母结点以及父母以上的祖辈结点
    # 输入：
    # net:array类型，网络结构，右上角元素ij表示各个连接边
    #     取值0表示无边，取值1表示Xi->Xj,取值-1表示Xi<-Xj
    # 输出：
    # parents：list类型，存储各个结点的父节点编号，用取值1~N代表各个节点
    # grands:list类型，存储各个结点更深的依赖节点，可以看成是“祖结点”
    # circle:list类型，存储环节点编号，若图中存在环，那么这个结点将是它本身的“祖结点”

    N = len(net)  # 节点数
    # -----确定父结点-----
    parents = [list(np.where(net[i, :] == -1)[0] + 1) +
               list(np.where(net[:, i] == 1)[0] + 1)
               for i in range(N)]
    grands = []
    # -----确定“祖结点”-----
    for i in range(N):
        grand = []
        # ---爷爷辈---
        for j in parents[i]:
            for k in parents[j - 1]:
                if k not in grand:
                    grand.append(k)
        # ---曾祖及以上辈---
        loop = True
        while loop:
            loop = False
            for j in grand:
                for k in parents[j - 1]:
                    if k not in grand:
                        grand.append(k)
                        loop = True
        grands.append(grand)
    # -----判断环结点-----
    circle = [i + 1 for i in range(N) if i + 1 in grands[i]]
    return parents, grands, circle


def draw(net, name=None, title=''):
    # 绘制贝叶斯网络的变量关系图
    # net:array类型，网络结构，右上角元素ij表示各个连接边
    #     取值0表示无边，取值1表示Xi->Xj,取值-1表示Xi<-Xj
    # name:指定各个节点的名称，默认为None，用x1,x2...xN作为名称
    N = net.shape[0]
    Level = np.ones(N, dtype=int)
    # -----确定层级-----
    for i in range(N):
        for j in range(i + 1, N):
            if net[i][j] == 1 and Level[j] <= Level[i]:
                Level[j] += 1
            if net[i][j] == -1 and Level[i] <= Level[j]:
                Level[i] += 1
    # -----确定横向坐标-----
    position = np.zeros(N)
    for i in set(Level):
        num = sum(Level == i)
        position[Level == i] = np.linspace(-(num - 1) / 2, (num - 1) / 2, num)
    # -----画图-----
    plt.figure()
    plt.title(title)
    # 设置出图显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # --先画出各个结点---
    for i in range(N):
        if name == None:
            text = 'x%d' % (i + 1)
        else:
            text = name[i]
        plt.annotate(text, xy=[position[i], Level[i]], bbox={'boxstyle': 'circle', 'fc': '1'}, ha='center')
    # --再画连接线---
    for i in range(N):
        for j in range(i + 1, N):
            if net[i][j] == 1:
                xy = np.array([position[j], Level[j]])
                xytext = np.array([position[i], Level[i]])
            if net[i][j] == -1:
                xy = np.array([position[i], Level[i]])
                xytext = np.array([position[j], Level[j]])
            if net[i][j] != 0:
                L = np.sqrt(sum((xy - xytext) ** 2))
                xy = xy - (xy - xytext) * 0.2 / L
                xytext = xytext + (xy - xytext) * 0.2 / L
                if (xy[0] == xytext[0] and abs(xy[1] - xytext[1]) > 1) or \
                        (xy[1] == xytext[1] and abs(xy[0] - xytext[0]) > 1):
                    arrowprops = dict(arrowstyle='->', connectionstyle='arc3,rad=0.3')
                    # 画弧线，避免遮挡(只考虑了横向和竖向边，暂未考虑斜向边遮挡的情况)
                else:
                    arrowprops = dict(arrowstyle='->')
                plt.annotate('', xy=xy, xytext=xytext, arrowprops=arrowprops, ha='center')
    plt.axis([position.min(), position.max(), Level.min(), Level.max() + 0.5])
    plt.axis('off')
    plt.show()


def coder(StateNums):
    # 编码器,
    # 设有结点x1,x2,...xN,各个结点的状态数为s1,s2,...sN,
    # 那么结点取值的组合数目为s1*s2*...sN,
    # 这些组合状态可以编码表示为[0,0,...0]~[s1-1,s2-1,...sN-1]
    # 输出：
    #     StateNums：各个结点状态数,
    #                比如[2,3,2]意为x1,x2,x3分别有2,3,2种状态，
    #                组合起来便有12种状态。
    # 输出：
    #     codes:用于遍历所有状态的索引编号，
    #          比如，对于[2,3],总共6种组合状态，遍历这6种组合状态的编码为：
    #          [0,0],[0,1],[0,2],[1,0],[1,1],[1,2]

    Nodes = len(StateNums)  # 结点数
    states = np.prod(StateNums)  # 组合状态数
    codes = []
    for s in range(states):
        s0 = s
        code = []
        for step in range(Nodes - 1):
            wight = np.prod(StateNums[step + 1:])
            code.append(s0 // wight)
            s0 = s0 % wight
        code.append(s0)
        codes.append(code)
    return codes


def EM(net, D, ZStateNum, Try=1):
    # EM算法计算隐变量概率分布Q(z),这里仅考虑单个隐变量的简单情况
    # 输入：
    #     net:贝叶斯网络结构，以矩阵右上角元素表示连接关系，
    #         约定将隐变量排在最后一个。
    #     D:可观测变量数据集
    #     ZStateNum:隐变量状态数(离散取值数目)结果
    #     Try:尝试次数，由于EM算法收敛到的结果受初始值影响较大，
    #         因此，尝试不同初始值，最终选择边际似然最大的。
    # 输出：
    #     Qz：隐变量概率分布

    # =====网络性质=====
    parents = [list(np.where(net[i, :] == -1)[0]) +
               list(np.where(net[:, i] == 1)[0])
               for i in range(len(net))]  # 计算各个结点的父节点
    # =====可观测变量参数=====
    m, Nx = D.shape  # 样本数和可观测变量数
    values = [np.unique(D[:, i]) for i in range(Nx)]  # 可观测变量的离散取值
    # =====隐变量子节点=====
    Zsonindex = list(np.where((net[:Nx, Nx:] == -1).any(axis=1))[0])  # 隐结点子节点索引号

    # =====运行多次EM，每次随机初始化Qz，最终选择边际似然最大的结果=====
    for t in range(Try):
        # =====隐变量分布初始化=====
        Qz = np.random.rand(m, ZStateNum)  # 初始化隐变量概率分布
        Qz = Qz / Qz.sum(axis=1).reshape(-1, 1)  # 概率归一化
        # Qz=np.c_[np.ones([m,1]),np.zeros([m,2])]
        # =====迭代更新Qz=====
        dif = 1  # 两次Qz的差别
        while dif > 1E-8:
            NewQz = np.ones(Qz.shape)
            # -----对于隐结点-----
            pa = parents[-1]  # 隐结点的父结点
            if len(parents[-1]) == 0:  # 如果隐结点没有父节点
                NewQz *= Qz.sum(axis=0)
            else:
                ValueNums = [len(values[p]) for p in pa]  # 各个父结点的状态数
                codes = coder(ValueNums)  # 用于遍历各种取值的编码
                for code in codes:
                    # 父结点取值组合
                    CombValue = [values[pa[p]][code[p]] for p in range(len(pa))]
                    index = np.where((D[:, pa] == CombValue).all(axis=1))[0]
                    NewQz[index] *= Qz[index].sum(axis=0) if len(index) != 0 else 1
            # -----对于隐结点的子结点-----
            for son in Zsonindex:
                # 分子部分
                pa = parents[son]  # 父结点,
                Nodes = pa + [son]  # 加上该结点本身,
                Nodes.remove(Nx)  # 然后，移去隐结点作为考察结点
                ValueNums = [len(values[N]) for N in Nodes]  # 各个结点的状态数
                codes = coder(ValueNums)  # 用于遍历各种取值的编码
                for code in codes:
                    CombValue = [values[Nodes[N]][code[N]] for N in range(len(Nodes))]
                    index = np.where((D[:, Nodes] == CombValue).all(axis=1))[0]
                    NewQz[index] *= Qz[index].sum(axis=0) if len(index) != 0 else 1
                # 分母部分
                pa = parents[son] + []  # 仅考察父结点
                pa.remove(Nx)  # 移去隐结点
                if len(pa) == 0:  # 如果父结点只有隐结点一个
                    NewQz /= Qz.sum(axis=0)
                else:
                    ValueNums = [len(values[p]) for p in pa]  # 各个父结点的状态数
                    codes = coder(ValueNums)  # 用于遍历各种取值的编码
                    for code in codes:
                        # 父结点取值组合
                        CombValue = [values[pa[p]][code[p]] for p in range(len(pa))]
                        index = np.where((D[:, pa] == CombValue).all(axis=1))[0]
                        NewQz[index] /= Qz[index].sum(axis=0) + 1E-100 if len(index) != 0 else 1
            NewQz = NewQz / NewQz.sum(axis=1).reshape(-1, 1)  # 归一化
            dif = np.sum((Qz - NewQz) ** 2, axis=1).mean()  # 新旧Qz的差别
            Qz = NewQz  # 更新Qz

        if t == 0:
            BestQz = Qz
            maxLL = LL(net, D, Qz, consider=(Zsonindex + [Nx]))
        else:
            newLL = LL(net, D, Qz, consider=(Zsonindex + [Nx]))
            if newLL > maxLL:
                maxLL = newLL
                BestQz = Qz
    return BestQz


def LL(net, D, Qz, consider=None):
    # 含有单个隐变量的情况下，计算边际似然
    # 输入：
    #     net:贝叶斯网络结构，以矩阵右上角元素表示连接关系，
    #         约定将隐变量排在最后一个。
    #     D:可观测变量数据集
    #     Qz：隐变量概率分布
    #     consider:所考察的结点。根据分析，
    #              边际似然中部分项可以表示为各个结点求和的形式，
    #              因此可以指定求和所包含的结点
    # 输出：
    #     LL：边际似然

    # =====网络性质=====
    parents = [list(np.where(net[i, :] == -1)[0]) +
               list(np.where(net[:, i] == 1)[0])
               for i in range(len(net))]  # 计算各个结点的父节点
    # =====可观测变量参数=====
    m, Nx = D.shape  # 样本数和可观测变量数
    values = [np.unique(D[:, i]) for i in range(Nx)]  # 可观测变量的离散取值
    # =====待考察结点=====
    if consider is None:
        consider = range(Nx + 1)
    # =====计算边际似然的求和项=====
    LL = 0
    # print(consider)
    for i in consider:
        # print(i)
        pa = parents[i]  # 父结点
        sign = 1
        for nodes in [pa + [i], pa]:  # nodes为当前所考察的结点
            if len(nodes) == 0:  # 考虑当前xi没有父结点的情况
                LL += sign * m * np.log(m)
                continue
            zin = nodes.count(Nx) != 0  # 是否含有隐结点
            if zin:
                nodes.remove(Nx)
            if len(nodes) == 0:  # 除了隐结点外没有其他结点
                mz = Qz.sum(axis=0)
                LL += sign * sum(np.log(mz ** mz))
            else:
                StateNums = [len(values[nd]) for nd in nodes]
                for code in coder(StateNums):
                    CombValue = [values[nodes[N]][code[N]] for N in range(len(nodes))]
                    index = np.where((D[:, nodes] == CombValue).all(axis=1))[0]
                    if zin:
                        mz = Qz[index].sum(axis=0)
                        LL += sign * sum(np.log(mz ** mz))
                    else:
                        mz = len(index)
                        LL += sign * (np.log(mz ** mz))
            sign *= -1
    # =====计算隐变量概率分布项=====
    LL -= np.sum(np.log(Qz ** Qz))
    return LL


def BIC(net, D, Qz, alpha=1, consider=None):
    # 计算BIC评分
    # 输入：
    #     net:贝叶斯网络结构，以矩阵右上角元素表示连接关系，
    #         约定将隐变量排在最后一个。
    #     D:可观测变量数据集
    #     Qz：隐变量分布
    #     alpha:结构项的比重系数
    #     consider:所考察的结点。根据分析，
    #              BIC评分中前两项可以表示为各个结点求和的形式，
    #              因此可以指定求和所包含的结点
    # 输出：
    #     np.array([struct,emp])：BIC评分结构项和经验项两部分的分值

    # -----从数据集D中提取一些信息-----
    m, Nx = D.shape  # 样本数和可观测变量数特征数
    values = [list(np.unique(D[:, i])) for i in range(len(D[0]))]  # 各个离散属性的可能取值
    values.append(range(ZStateNum))  # 再加上隐变量的取值数
    # -----父节点-----
    parents = [list(np.where(net[i, :] == -1)[0]) +
               list(np.where(net[:, i] == 1)[0])
               for i in range(len(net))]
    # -----计算BIC评分-----
    emp = -LL(net, D, Qz, consider)  # 经验项部分调用LL函数来计算
    struct = 0  # 下面计算结构项
    if consider is None:
        consider = range(Nx + 1)
    for i in consider:
        r = len(values[i])  # Xi结点的取值数
        pa = parents[i]  # Xi的父节点编号
        nums = [len(values[p]) for p in pa]  # 父节点取值数
        q = np.prod(nums)  # 父节点取值组合数
        struct += q * (r - 1) / 2 * np.log(m)  # 对结构项的贡献
    return np.array([struct * alpha, emp])


def BIC_change(net0, D, Qz, change, alpha=1):
    # 计算贝叶斯网络结构发生变化后BIC评分的变化量
    # 输入：net0:变化前的网络结构
    #       D:数据集
    #       Qz:隐结点分布
    #      change:网络结构的变化，内容为[i,j,value],
    #             意为xi到xj之间的连接边变为value值
    #      alpha：计算BIC评分时，结构项的比重系数
    # 输出：dscore:BIC评分的改变,内容为[struct,emp],
    #             分别表示结构项和经验项的变化
    #       NewQz:新的隐结点分布

    # =====网络结构的改变
    i, j, value = change
    consider = [i, j]
    net1 = net0.copy()
    net1[i, j] = value
    # =====隐变量子节点
    son0 = list(np.where(net0[:-1, -1] == -1)[0])
    son1 = list(np.where(net1[:-1, -1] == -1)[0])
    # =====判断是否需要重新运行EM
    if j == len(net0) - 1 or (i in son0) or (j in son0):
        Qz1 = EM(net1, D, Qz.shape[1], 12)
        consider = consider + son0 + son1 + [len(net0) - 1]
        consider = np.unique(consider)
    else:
        Qz1 = Qz
    dscore = BIC(net1, D, Qz1, alpha, consider) - BIC(net0, D, Qz, alpha, consider)
    return dscore, Qz1


# ================================================
#                  主程序
# ================================================

# ==============西瓜数据集2.0======================
# 将X和类标记Y放一起
D = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
     ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
     ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
     ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
     ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
     ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
     ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
     ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
     ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
     ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
     ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
     ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
     ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
     ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
     ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
     ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
     ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']]
D = np.array(D)
FeatureName = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']

# ======将“脐部”视为隐变量，对数据进行相应的修改=====
D = D[:, [0, 1, 2, 3, 5, 6]]  # 可观测数据集
XName = ['色泽', '根蒂', '敲声', '纹理', '触感', '好瓜']  # x变量名称
ZName = ['脐部']  # 隐变量名称
ZStateNum = 3  # 隐变量的状态数(离散取值数目)
FeatureName = XName + ZName  # 包括可观测变量和隐变量的所有变量的名称

# =================初始化为朴素贝叶斯网=============

# 构建贝叶斯网络，右上角元素ij表示各个连接边
# 取值0表示无边，取值1表示Xi->Xj,取值-1表示Xi<-Xj
m = D.shape[0]  # 样本数
N = len(XName) + 1  # 结点数

net = np.zeros((N, N))
choose = 4  # 选择初始化类型，可选1,2,3,4
# 分别代表全独立网络、朴素贝叶斯网络、全连接网络,随机网络
if choose == 1:  # 全独立网络:图中没有任何连接边
    pass
elif choose == 2:  # 朴素贝叶斯网络:所有其他特征的父节点都是类标记"好瓜"
    net[:-1, -1] = -1
elif choose == 3:  # 全连接网络：任意两个节点都有连接边
    again = True  # 若存在环，则重新生成
    while again:
        for i in range(N - 1):
            net[i, i + 1:] = np.random.randint(0, 2, N - i - 1) * 2 - 1
        again = len(relationship(net)[2]) != 0
elif choose == 4:  # 随机网络：任意两个节点之间的连接边可有可无
    again = True  # 若存在环，则重新生成
    while again:
        for i in range(N - 1):
            net[i, i + 1:] = np.random.randint(-1, 2, N - i - 1)
        again = len(relationship(net)[2]) != 0

draw(net, FeatureName, '初始网络结构')

# ==============下山法搜寻BIC下降的贝叶斯网==========
alpha = 0.1  # BIC评分的结构项系数
Qz = EM(net, D, ZStateNum, 12)
score0 = BIC(net, D, Qz, alpha)
score = [score0]
print('===========训练贝叶斯网============')
print('初始BIC评分:%.3f(结构%.3f,经验%.3f)' % (sum(score0), score0[0], score0[1]))
eta, tao = 0.1, 50  # 允许eta的概率调整到BIC评分增大的网络
# 阈值随迭代次数指数下降
for loop in range(500):
    # 随机指定需要调整的连接边的两个节点：Xi和Xj
    i, j = np.random.randint(0, N, 2)
    while i == j:
        i, j = np.random.randint(0, N, 2)
    i, j = (i, j) if i < j else (j, i)
    # 确定需要调整的结果
    value0 = net[i, j]  # 可能为0,1,-1
    change = np.random.randint(2) * 2 - 1  # 结果+1或-1
    value1 = (value0 + 1 + change) % 3 - 1  # 调整后的取值
    net1 = net.copy()
    net1[i, j] = value1
    if value1 != 0 and len(relationship(net1)[2]) != 0:
        # 如果value1取值非零，说明为转向或者增边
        # 若引入环，则放弃该调整
        continue
    chage, NewQz = BIC_change(net, D, Qz, [i, j, value1], alpha)
    if sum(chage) < 0 or np.random.rand() < eta * np.exp(-loop / tao):
        score0 = score0 + chage
        score.append(score0)
        net = net1
        Qz = NewQz
        print('调整后BIC评分:%.3f(结构%.3f,经验%.3f)'
              % (sum(score0), score0[0], score0[1]))
    else:
        continue

draw(net, FeatureName, '最终网络结构,alpha=%.3f' % (alpha))
Qz = EM(net, D, ZStateNum)

score = np.array(score)
x = np.arange(len(score))
plt.title('BIC贝叶斯网络结构搜索过程,alpha=%.3f' % (alpha))
plt.xlabel('更新次数')
plt.ylabel('分值')
plt.plot(x, score[:, 0], '.r-')
plt.plot(x, score[:, 1], '.b-')
plt.plot(x, score.sum(axis=1), '.k-')
plt.legend(['struct', 'emp', 'BIC(struct+emp)'])
plt.show()
