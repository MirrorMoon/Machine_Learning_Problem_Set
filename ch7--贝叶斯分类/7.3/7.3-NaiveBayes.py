import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple

# 训练朴素贝叶斯模型
def train_nb(X, y):
    m, n = X.shape  # 样本数 m 和特征数 n
    #书式7.19
    p1 = (len(y[y == '是']) + 1) / (m + 2)  # 使用拉普拉斯平滑处理正例的先验概率

    p1_list, p0_list = [], []  # 分别保存正例和负例下的条件概率

    X1, X0 = X[y == '是'], X[y == '否']  # 提取正例和负例样本
    m1, m0 = X1.shape[0], X0.shape[0]  # 正例和负例样本数量

    for i in range(n):  # 遍历每个特征
        #语法是取所有行的第i列
        xi = X.iloc[:, i]  # 提取第 i 个特征
        p_xi = namedtuple(X.columns[i], ['is_continuous', 'conditional_pro'])

        # 判断是否为连续特征
        is_continuous = type_of_target(xi) == 'continuous'
        xi1, xi0 = X1.iloc[:, i], X0.iloc[:, i]  # 取出第 i 个特征在正例和负例中的数据

        if is_continuous:  # 处理连续特征
            #分别计算正例和负例下特征xi均值和方差，因为对于连续特征而言，其条件概率就将ci类样本在xi上的均值与方差带入正态分布的概率密度函数
            xi1_mean, xi1_var = np.mean(xi1), np.var(xi1)
            xi0_mean, xi0_var = np.mean(xi0), np.var(xi0)
            p1_list.append(p_xi(is_continuous, [xi1_mean, xi1_var]))
            p0_list.append(p_xi(is_continuous, [xi0_mean, xi0_var]))
        else:  # 处理离散特征
            unique_value = xi.unique()  # 该特征的唯一取值
            nvalue = len(unique_value)

            # 使用拉普拉斯平滑计算频率，这两句代码即7.20式中的分子
            #reindex是因为正例或者负例可能不包含所有的特征取值，因此需要填充0
            xi1_value_count = pd.Series(xi1).value_counts().reindex(unique_value, fill_value=0) + 1
            xi0_value_count = pd.Series(xi0).value_counts().reindex(unique_value, fill_value=0) + 1
            #取对数将联合概率分布由乘法转为加法，避免下溢
            p1_list.append(p_xi(is_continuous, np.log(xi1_value_count / (m1 + nvalue))))
            p0_list.append(p_xi(is_continuous, np.log(xi0_value_count / (m0 + nvalue))))

    #分别返回正例的先验概率和特征的条件概率，以及负例的条件概率
    return p1, p1_list, p0_list

# 预测函数
def predict_nb(x, p1, p1_list, p0_list):
    n = len(x)  # 特征数
    x_p1, x_p0 = np.log(p1), np.log(1 - p1)  # 初始化正例和负例的对数概率,用来相加

    for i in range(n):  # 遍历每个特征，分别获取其条件概率，对于离散属性已经将其转为对数，而对于连续属性，由于要带入正态分布，因此在这里转为对数
        p1_xi, p0_xi = p1_list[i], p0_list[i]

        if p1_xi.is_continuous:  # 连续特征使用正态分布计算概率
            mean1, var1 = p1_xi.conditional_pro
            mean0, var0 = p0_xi.conditional_pro
            #这里就是正态分布的密度公式
            x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2)))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2)))
        else:  # 离散特征直接累加条件概率的对数值
            x_p1 += p1_xi.conditional_pro[x[i]]
            x_p0 += p0_xi.conditional_pro[x[i]]

    return '是' if x_p1 > x_p0 else '否'  # 取概率最大的那个作为预测结果，也就是最大后验概率准则

# 主函数
if __name__ == '__main__':
    data_path = r'D:\Machine_Learning_Problem_Set\data\watermelon3_0_Ch.csv'
    data = pd.read_csv(data_path, index_col=0)

    X, y = data.iloc[:, :-1], data.iloc[:, -1]  # 分割特征矩阵 X 和标签 y
    p1, p1_list, p0_list = train_nb(X, y)  # 训练模型

    x_test = X.iloc[0, :]  # 选择测试样本
    print(predict_nb(x_test, p1, p1_list, p0_list))  # 输出预测结果
