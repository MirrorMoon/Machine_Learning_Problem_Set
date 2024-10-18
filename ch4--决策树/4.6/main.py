'''
treeCreater 和 treePlotter 代码见 ch4/4.3-4.4
数据量不大，不同的随机数种子，测试集的准确率变化较大
'''
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

import treeCreater
from scipy.stats import ttest_rel
def cross_val_score(estimator, X, y,X_val,y_val,cv=10):
    # 自己写一个 k-10 交叉验证的实现

    X_tem=X.reset_index(drop=True)
    y_tem=y.reset_index(drop=True)
    X_val_tem = X_val.reset_index(drop=True)
    y_val_tem = y_val.reset_index(drop=True)
    m = X_tem.shape[0]  # 样本数量
    num_split = int(m / cv)  # 计算每折的样本数量
    score_my = []  # 初始化自定义交叉验证的分数列表
    for i in range(cv):  # 遍历每一折
        test_index = range(i * num_split, (i + 1) * num_split)  # 确定测试集的索引范围
        X_test_ = X_tem.iloc[test_index]  # 获取测试集特征
        y_test_ = y_tem.iloc[test_index]  # 获取测试集标签

        # 获取训练集特征和标签，去掉测试集样本
        X_train_ = X_tem.drop(index=test_index)
        y_train_ = y_tem.drop(index=test_index)
        # 训练模型
        estimator.fit(X_train_, y_train_,X_val_tem,y_val_tem)

        # 将当前折的分数添加到列表
        score_my.append(estimator.score(X_test_, y_test_))
    return score_my
# 加载 Iris 数据集
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.Series(iris['target_names'][iris['target']])
#
# # 取 30 个样本为测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
#
# 剩下 120 个样本中，取 30 个作为剪枝时的验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=15)

# 不剪枝
tree_no_pruning = treeCreater.DecisionTree('gini')
# 使用交叉验证
no_pruning_scores = cross_val_score(tree_no_pruning, X_train, y_train,X_val,y_val,cv=5)  # 5 折交叉验证
print('不剪枝：', np.mean(no_pruning_scores))
accuracy_no_pruning = np.mean(no_pruning_scores)

# 预剪枝
tree_pre_pruning = treeCreater.DecisionTree('gini', 'pre_pruning')
# 使用交叉验证
pre_pruning_scores = cross_val_score(tree_pre_pruning, X_train, y_train,X_val,y_val,cv=5)
print('预剪枝：', np.mean(pre_pruning_scores))
accuracy_pre_pruning = np.mean(pre_pruning_scores)

# 后剪枝
tree_post_pruning = treeCreater.DecisionTree('gini', 'post_pruning')
# 使用交叉验证
post_pruning_scores = cross_val_score(tree_post_pruning, X_train, y_train,X_val,y_val,cv=5)
print('后剪枝：', np.mean(post_pruning_scores))
accuracy_post_pruning = np.mean(post_pruning_scores)

# 输出准确率
print(f"未剪枝模型平均准确率: {accuracy_no_pruning:.2f}")
print(f"预剪枝模型平均准确率: {accuracy_pre_pruning:.2f}")
print(f"后剪枝模型平均准确率: {accuracy_post_pruning:.2f}")

# 可以根据需要继续进行 t 检验等
# 进行配对 t 检验
t_stat_no_vs_pre, p_value_no_vs_pre = ttest_rel(no_pruning_scores, pre_pruning_scores)
t_stat_no_vs_post, p_value_no_vs_post = ttest_rel(no_pruning_scores, post_pruning_scores)
t_stat_pre_vs_post, p_value_pre_vs_post = ttest_rel(pre_pruning_scores, post_pruning_scores)

# 输出 t 检验结果
# 定义显著性水平
alpha = 0.05

# 输出未剪枝 vs 预剪枝结果
if p_value_no_vs_pre < alpha:
    significance_no_vs_pre = "显著"
else:
    significance_no_vs_pre = "不显著"

print(f"未剪枝 vs 预剪枝: t-statistic = {t_stat_no_vs_pre:.4f}, p-value = {p_value_no_vs_pre:.4f} ({significance_no_vs_pre})")

# 输出未剪枝 vs 后剪枝结果
if p_value_no_vs_post < alpha:
    significance_no_vs_post = "显著"
else:
    significance_no_vs_post = "不显著"

print(f"未剪枝 vs 后剪枝: t-statistic = {t_stat_no_vs_post:.4f}, p-value = {p_value_no_vs_post:.4f} ({significance_no_vs_post})")

# 输出预剪枝 vs 后剪枝结果
if p_value_pre_vs_post < alpha:
    significance_pre_vs_post = "显著"
else:
    significance_pre_vs_post = "不显著"

print(f"预剪枝 vs 后剪枝: t-statistic = {t_stat_pre_vs_post:.4f}, p-value = {p_value_pre_vs_post:.4f} ({significance_pre_vs_post})")




# treePlottter.create_plot(tree_no_pruning.tree_)


