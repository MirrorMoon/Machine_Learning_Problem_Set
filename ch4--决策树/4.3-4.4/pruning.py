import pandas as pd
import numpy as np


def post_pruning(X_train, y_train, X_val, y_val, tree_=None):
    if tree_.is_leaf:
        return tree_

    if X_val.empty:         # 验证集为空集时，不再剪枝
        return tree_

    #获取当前节点样本数最多的类别
    most_common_in_train = pd.value_counts(y_train).index[0]
    current_accuracy = np.mean(y_val == most_common_in_train)  # 当前节点下验证集样本准确率（未划分）

    if tree_.is_continuous:
        up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
        up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
        down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

        up_subtree = post_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                  y_val[up_part_val],
                                  tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
        tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree
        down_subtree = post_pruning(X_train[down_part_train], y_train[down_part_train],
                                    X_val[down_part_val], y_val[down_part_val],
                                    tree_.subtree['< {:.3f}'.format(tree_.split_value)])
        tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree

        tree_.high = max(up_subtree.high, down_subtree.high) + 1
        tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

        if up_subtree.is_leaf and down_subtree.is_leaf:
            def split_fun(x):
                if x >= tree_.split_value:
                    return '>= {:.3f}'.format(tree_.split_value)
                else:
                    return '< {:.3f}'.format(tree_.split_value)

            val_split = X_val.loc[:, tree_.feature_name].map(split_fun)
            right_class_in_val = y_val.groupby(val_split).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            if current_accuracy > split_accuracy:  # 若当前节点为叶节点时的准确率大于不剪枝的准确率，则进行剪枝操作——将当前节点设为叶节点
                set_leaf(pd.value_counts(y_train).index[0], tree_)
    else:
        max_high = -1
        tree_.leaf_num = 0
        is_all_leaf = True  # 判断当前节点下，所有子树是否都为叶节点

        for key in tree_.subtree.keys():
            #筛选出当前特征之下所有特征值为key的数据
            this_part_train = X_train.loc[:, tree_.feature_name] == key
            this_part_val = X_val.loc[:, tree_.feature_name] == key
            #因为进行剪枝的话只会涉及到叶子节点和分支节点的合并，所以对于全是叶子节点分支节点，即便其不在最底层，也可以先剪枝判断，显然这与从最底层往上判断时一致的
            tree_.subtree[key] = post_pruning(X_train[this_part_train], y_train[this_part_train],
                                              X_val[this_part_val], y_val[this_part_val], tree_.subtree[key])
            if tree_.subtree[key].high > max_high:
                max_high = tree_.subtree[key].high
            tree_.leaf_num += tree_.subtree[key].leaf_num

            if not tree_.subtree[key].is_leaf:
                is_all_leaf = False
        tree_.high = max_high + 1

        if is_all_leaf:  # 若所有子节点都为叶节点，则考虑是否进行剪枝
            #将验证集分组后，判断每组中样本的类别是否与决策树叶子节点划定的一致
            right_class_in_val = y_val.groupby(X_val.loc[:, tree_.feature_name]).apply(
                lambda x: np.sum(x == tree_.subtree[x.name].leaf_class))
            split_accuracy = right_class_in_val.sum() / y_val.shape[0]

            if current_accuracy > split_accuracy:  # 若当前节点为叶节点时的准确率大于划分后的准确率，则进行剪枝操作——将当前节点设为叶节点
                set_leaf(pd.value_counts(y_train).index[0], tree_)

    return tree_


#从根节点开始向下剪枝
def pre_pruning(X_train, y_train, X_val, y_val, tree_=None):

    if tree_.is_leaf:  # 若当前节点已经为叶节点，那么就直接return了
        return tree_

    if X_val.empty: # 验证集为空集时，不再剪枝（剪枝只可能发生在能产生分支的情况下）
        return tree_
    # 在计算准确率时，由于西瓜数据集的原因，好瓜和坏瓜的数量会一样，这个时候选择训练集中样本最多的类别时会不稳定（因为都是50%），
    # 导致准确率不稳定，当然在数量大的时候这种情况很少会发生。
    #value_counts返回的列表是降序列表
    most_common_in_train = pd.value_counts(y_train).index[0]
    #计算验证集精度
    current_accuracy = np.mean(y_val == most_common_in_train)

    if tree_.is_continuous:  # 连续值时，需要将样本分割为两部分，来计算分割后的正确率

        #生成决策树时，node里面就已经保存了最优分割点
        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val,
                                                  split_value=tree_.split_value)

        if current_accuracy >= split_accuracy:  # 当前节点为叶节点时准确率大于分割后的准确率时，选择不划分
            set_leaf(pd.value_counts(y_train).index[0], tree_)

        else:
            #分别根据分割点获取训练集和验证集的两个部分
            up_part_train = X_train.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_train = X_train.loc[:, tree_.feature_name] < tree_.split_value
            up_part_val = X_val.loc[:, tree_.feature_name] >= tree_.split_value
            down_part_val = X_val.loc[:, tree_.feature_name] < tree_.split_value

            #对>=对应分支继续递归进行剪枝
            up_subtree = pre_pruning(X_train[up_part_train], y_train[up_part_train], X_val[up_part_val],
                                     y_val[up_part_val],
                                     tree_.subtree['>= {:.3f}'.format(tree_.split_value)])
            #剪枝结束更新决策树
            tree_.subtree['>= {:.3f}'.format(tree_.split_value)] = up_subtree
            #同上
            down_subtree = pre_pruning(X_train[down_part_train], y_train[down_part_train],
                                       X_val[down_part_val],
                                       y_val[down_part_val],
                                       tree_.subtree['< {:.3f}'.format(tree_.split_value)])
            tree_.subtree['< {:.3f}'.format(tree_.split_value)] = down_subtree
            #最后更新
            tree_.high = max(up_subtree.high, down_subtree.high) + 1
            tree_.leaf_num = (up_subtree.leaf_num + down_subtree.leaf_num)

    else:  # 若是离散值，则遍历所有值，计算分割后正确率

        split_accuracy = val_accuracy_after_split(X_train[tree_.feature_name], y_train,
                                                  X_val[tree_.feature_name], y_val)
        #如果未划分的精度>=划分后的精度，那么就意味着不需要划分，并将当前节点设置未叶子节点，并取当前集合中样本数最多的类别作为结果
        if current_accuracy >= split_accuracy:
            set_leaf(pd.value_counts(y_train).index[0], tree_)

        else:
            max_high = -1
            tree_.leaf_num = 0
            #获取当前节点的分支
            for key in tree_.subtree.keys():
                #获取当前分支需要进行判断的训练集和验证集
                this_part_train = X_train.loc[:, tree_.feature_name] == key
                this_part_val = X_val.loc[:, tree_.feature_name] == key
                #利用dfs对每个分支做同样的预剪枝
                tree_.subtree[key] = pre_pruning(X_train[this_part_train], y_train[this_part_train],
                                                 X_val[this_part_val],
                                                 y_val[this_part_val], tree_.subtree[key])
                #获取对当前分支剪枝完成后的子树高度，并且遍历获取最大者
                if tree_.subtree[key].high > max_high:
                    max_high = tree_.subtree[key].high
                #计算剪枝后的叶子节点个数
                tree_.leaf_num += tree_.subtree[key].leaf_num
            #得到剪枝后的决策树高度
            tree_.high = max_high + 1
    return tree_


def set_leaf(leaf_class, tree_):
    # 设置节点为叶节点
    tree_.is_leaf = True  # 若划分前正确率大于划分后正确率。则选择不划分，将当前节点设置为叶节点
    tree_.leaf_class = leaf_class
    tree_.feature_name = None
    tree_.feature_index = None
    tree_.subtree = {}
    tree_.impurity = None
    tree_.split_value = None
    tree_.high = 0  # 重新设立高 和叶节点数量
    tree_.leaf_num = 1

#计算分割后的精度
def val_accuracy_after_split(feature_train, y_train, feature_val, y_val, split_value=None):
    # 若是连续值时，需要需要按切分点对feature 进行分组，若是离散值，则不用处理
    if split_value is not None:
        #这里是生成可视化字符串
        def split_fun(x):
            if x >= split_value:
                return '>= {:.3f}'.format(split_value)
            else:
                return '< {:.3f}'.format(split_value)

        #对feature_train中的元素利用函数split_fun完成映射，得到字符串,这样我们就可以根据字符串进行分组
        train_split = feature_train.map(split_fun)
        val_split = feature_val.map(split_fun)

    else:
        train_split = feature_train
        val_split = feature_val

    #对训练集标签根据特征值进行分组，并计算出每个分组（特征值）样本最多的类别（好瓜坏瓜）
    majority_class_in_train = y_train.groupby(train_split).apply(
        lambda x: pd.value_counts(x).index[0])  # 计算各特征下样本最多的类别
    # 对验证集标签根据特征值进行分组，并计算出每个分组与训练集给出的结果一致的个数（即对每个特征值求出判断正确的样本数）
    right_class_in_val = y_val.groupby(val_split).apply(
        lambda x: np.sum(x == majority_class_in_train[x.name]))  # 计算各类别对应的数量

    return right_class_in_val.sum() / y_val.shape[0]  # 返回准确率
