# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:02:12 2020

@author: lsly
"""
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


def BIC_score(net, D, consider=None):
    # 计算评分函数
    # 输入：
    #     net:贝叶斯网络
    #     D:数据集
    # 输出：
    #    [struct,emp]:评分函数的结构项和经验项

    # -----从数据集D中提取一些信息-----
    m, N = D.shape  # 样本数和特征数
    values = [np.unique(D[:, i]) for i in range(len(D[0]))]  # 各个离散属性的可能取值
    # -----父节点-----
    parents = [list(np.where(net[i, :] == -1)[0] + 1) +
               list(np.where(net[:, i] == 1)[0] + 1)
               for i in range(N)]
    # -----计算BIC评分-----
    struct, emp = 0, 0  # BIC评分的结构项和经验项
    if consider == None:
        consider = range(N)
    for i in consider:
        r = len(values[i])  # Xi结点的取值数
        pa = parents[i]  # Xi的父节点编号
        nums = [len(values[p - 1]) for p in pa]  # 父节点取值数
        q = np.prod(nums)  # 父节点取值组合数
        struct += q * (r - 1) / 2 * np.log(m)  # 对结构项的贡献
        # -----如果父节点数目为零，亦即没有父节点
        if len(pa) == 0:
            for value_k in values[i]:
                Dk = D[D[:, i] == value_k]  # D中Xi取值v_k的子集
                mk = len(Dk)  # Dk子集大小
                if mk > 0:
                    emp -= mk * np.log(mk / m)  # 对经验项的贡献
            continue
        # -----有父节点时，通过编码方式，遍历所有父节点取值组合
        for code in range(q):
            # 解码：比如，父节点有2×3种组合，
            # 将0~5解码为[0,0]~[1,2]
            code0 = code

            decode = []
            for step in range(len(pa) - 1):
                wight = np.prod(nums[step + 1:])
                decode.append(code0 // wight)
                code0 = code0 % wight
            decode.append(code0)

            # 父节点取某一组合时的子集
            index = range(m)  # 子集索引号，初始为全集D
            # 起初误将m写为N，该错误极不容易发现，两天后发现并更正
            for j in range(len(pa)):
                indexj = np.where(D[:, pa[j] - 1] == values[pa[j] - 1][decode[j]])[0]
                index = np.intersect1d(index, indexj)
            Dij = D[index, :]  # 子集
            mij = len(Dij)  # 子集大小
            if mij > 0:  # 仅当子集非空时才计算该种情况
                for value_k in values[i]:
                    Dijk = Dij[Dij[:, i] == value_k]  # Dij中Xi取值v_k的子集
                    mijk = len(Dijk)  # Dijk子集大小
                    if mijk > 0:
                        emp -= mijk * np.log(mijk / mij)  # 对经验项的贡献
    return np.array([struct, emp])


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

# =================初始化贝叶斯网结构=============

# 构建贝叶斯网络，右上角元素ij表示各个连接边
# 取值0表示无边，取值1表示Xi->Xj,取值-1表示Xi<-Xj
m, N = D.shape

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
score0 = BIC_score(net, D)
score = [score0]
print('===========训练贝叶斯网============')
print('初始BIC评分:%.3f(结构%.3f,经验%.3f)' % (sum(score0), score0[0], score0[1]))

eta, tao = 0.1, 50  # 允许eta的概率调整到BIC评分增大的网络
# 阈值随迭代次数指数下降
for loop in range(10000):
    # 随机指定需要调整的连接边的两个节点：Xi和Xj
    i, j = np.random.randint(0, N, 2)
    while i == j:
        i, j = np.random.randint(0, N, 2)
    i, j = (i, j) if i < j else (j, i)
    # 确定需要调整的结果
    value0 = net[i, j]
    change = np.random.randint(2) * 2 - 1  # 结果为+1或-1
    value1 = (value0 + 1 + change) % 3 - 1  # 调整后的取值
    net1 = net.copy()
    net1[i, j] = value1
    if value1 != 0 and len(relationship(net1)[2]) != 0:
        # 如果value1取值非零，说明为转向或者增边
        # 若引入环，则放弃该调整
        continue
    delta_score = BIC_score(net1, D, [i, j]) - BIC_score(net, D, [i, j])
    if sum(delta_score) < 0 or np.random.rand() < eta * np.exp(-loop / tao):
        score0 = score0 + delta_score
        score.append(score0)
        print('调整后BIC评分:%.3f(结构%.3f,经验%.3f)'
              % (sum(score0), score0[0], score0[1]))
        net = net1
    else:
        continue

draw(net, FeatureName, '最终网络结构')

score = np.array(score)
x = np.arange(len(score))
plt.title('BIC贝叶斯网络结构搜索过程')
plt.xlabel('更新次数')
plt.ylabel('分值')
plt.plot(x, score[:, 0], '.r-')
plt.plot(x, score[:, 1], '.b-')
plt.plot(x, score.sum(axis=1), '.k-')
plt.legend(['struct', 'emp', 'BIC(struct+emp)'])
plt.show()
