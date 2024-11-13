import numpy as np


def AODE(x, X, Y, laplace=True, mmin=3):
    # 平均独依赖贝叶斯分类器
    # x 待预测样本
    # X 训练样本特征值
    # Y 训练样本标记
    # laplace 是否采用“拉普拉斯修正”，默认为真
    # mmin 作为父属性最少需要的样本数
    #利用集合去重
    C = list(set(Y))  # 所有可能标记，计算先验概率
    N = [len(set([x[i] for x in X])) for i in range(len(x))]  # 各个属性的可能取值数
    p = []  # 存储概率值
    print('===============平均独依赖贝叶斯分类器(AODE)===============')
    for c in C:
        # --------求取类先验概率P(c)--------
        Xc = [X[i] for i in range(len(Y)) if Y[i] == c]  # c类样本子集
        pc = (len(Xc) + laplace) / (len(X) + laplace * len(C))  # 类先验概率，式7.19
        print('P(c=%s)=%.3f;' % (c, pc))
        # --------求取父属性概率P(c,xi)--------
        p_cxi = np.zeros(len(x))  # 将计算结果存储在一维向量p_cxi中
        for i in range(len(x)):#遍历每个特征
            if type(x[i]) != type(X[0][i]):
                print(
                    '测试样本数据第%d个特征的数据类型与训练样本数据类型不符,无法预测!' % (i + 1))
                return None
            if type(x[i]) == str:
                # 若为字符型特征，按离散取值处理
                Xci = [1 for xc in Xc if xc[i] == x[i]]  # c类子集中第i个特征取值为xi的样本
                # p_cxi[i] = (len(Xci) + laplace) / (len(X)
                #                                    + laplace * len(C) * N[i])
                p_cxi[i] = (len(Xci) + laplace) / (len(X)
                                                   + laplace * N[i])
            elif type(x[i]) == float:
                # 若为浮点型特征，按高斯分布处理
                # 获取c类样本子集中第i个特征的取值
                Xci = [xc[i] for xc in Xc]
                u = np.mean(Xci)
                #计算无偏估计的标准差
                sigma = np.std(Xci, ddof=1)
                #带入密度公式
                p_cxi[i] = pc / np.sqrt(2 * np.pi) / \
                           sigma * np.exp(-(x[i] - u) ** 2 / 2 / sigma ** 2)
            else:
                print('目前只能处理字符型和浮点型数据，对于其他类型有待扩展相应功能。')
                return None
        print(''.join(['p(c=%d,xi)=' % c] + ['%.3f' % p_cxi[i] +
                                             (lambda i: ';' if i == len(x) - 1 else ',')(i) for i in range(len(x))]))

        # --------求取父属性条件依赖概率P(xj|c,xi)--------
        p_cxixj = np.eye(len(x))  # 将计算结果存储在二维向量p_cxixj中
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    continue
                # ------根据xi和xj是离散还是连续属性分为多种情况-----
                #父子均为离散，直接套式7.25
                if type(x[i]) == str and type(x[j]) == str:
                    Xci = [xc for xc in Xc if xc[i] == x[i]]
                    #这里在上面已经取出(c且xi的基础上在施加xj
                    Xcij = [1 for xci in Xci if xci[j] == x[j]]
                    #式7.25
                    p_cxixj[i, j] = (len(Xcij) + laplace) / (len(Xci) + laplace * N[j])
                #父为离散，子为连续
                if type(x[i]) == str and type(x[j]) == float:
                    Xci = [xc for xc in Xc if xc[i] == x[i]]
                    Xcij = [xci[j] for xci in Xci]
                    # 若子集Dc,xi数目少于2个，则无法用于估计正态分布参数，
                    # 则将其设为标准正态分布
                    if len(Xci) == 0:
                        u = 0
                    else:
                        u = np.mean(Xcij)
                    if len(Xci) < 2:
                        sigma = 1
                    else:
                        sigma = np.std(Xcij, ddof=1)
                    p_cxixj[i, j] = 1 / np.sqrt(2 * np.pi) / \
                                    sigma * np.exp(-(x[j] - u) ** 2 / 2 / sigma ** 2)
                #父为连续，子为离散
                if type(x[i]) == float and type(x[j]) == str:
                    Xcj = [xc for xc in Xc if xc[j] == x[j]]
                    #取出满足子属性条件的父属性
                    Xcji = [xcj[i] for xcj in Xcj]
                    if len(Xcj) == 0:
                        u = 0
                    else:
                        u = np.mean(Xcji)
                    if len(Xcj) < 2:
                        sigma = 1
                    else:
                        sigma = np.std(Xcji, ddof=1)
                    p_cxixj[i, j] = 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x[i] - u) ** 2 / 2 / sigma ** 2) * p_cxi[
                        j] / p_cxi[i]
                #父子均为连续
                if type(x[i]) == float and type(x[j]) == float:
                    #同时取出来
                    Xcij = np.array([[xc[i], xc[j]] for xc in Xc])
                    u = Xcij.mean(axis=0).reshape(1, -1)
                    #因为是多维正态分布，所以这里从原来的样本减均值的平方变为了dot，同样的，也是采用无偏估计
                    sigma2 = (Xcij - u).T.dot(Xcij - u) / (Xcij.shape[0] - 1)
                    p_cxixj[i, j] = 1 / 2 * np.pi / np.sqrt(np.linalg.det(sigma2)) \
                                    * np.exp(-0.5 * ([x[i], x[j]] - u).
                                             dot(np.linalg.inv(sigma2)).
                                             dot(([x[i], x[j]] - u).T)) * pc / p_cxi[i]
        print(''.join([(lambda j: 'p(xj|c=%d,x%d)=' % (c, i + 1) if j == 0 else '')(j)
                       + '%.3f' % p_cxixj[i][j]
                       + (lambda i: ';\n' if i == len(x) - 1 else ',')(j)
                       for i in range(len(x)) for j in range(len(x))]))
        # --------求计算总的概率∑iP(c,xi)*∏jP(xj|c,xi)--------
        sump = 0
        #mmin是阈值m'
        for i in range(len(x)):
            #只有当前属性取值为xi的样本数>m'才纳入累加
            if len([1 for xi in X if xi[i] == x[i]]) >= mmin:
                sump += p_cxi[i] * p_cxixj[i, :].prod()
        print('P(c=%d,x) ∝ %.3E\n' % (c, sump))
        p.append(sump)
    # --------做预测--------
    predict = C[p.index(max(p))]
    print('===============预测结果：%s类===============' % predict)
    return predict


# ====================================
#              主程序
# ====================================
# 表4.3 西瓜数据集3.0
# FeatureName=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
# LabelName={1:'好瓜',0:'坏瓜'}
X = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
     ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376],
     ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264],
     ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318],
     ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215],
     ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237],
     ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149],
     ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211],
     ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091],
     ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267],
     ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057],
     ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099],
     ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161],
     ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198],
     ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370],
     ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042],
     ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103]]
Y = [1] * 8 + [0] * 9

x = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]  # 测试例"测1"
print("测1样本：")
AODE(x, X, Y)


# 任意设置一个新的"测例"
x = ['浅白', '蜷缩', '沉闷', '稍糊', '平坦', '硬滑', 0.5, 0.1]
print("\n任设的一个新测例，观察结果：")
AODE(x, X, Y)