'''
拟合函数中，X支持pd.DataFrame数据类型；y暂只支持pd.Series类型，其他数据类型未测试，
目前在西瓜数据集上和sklearn中自带的iris数据集上运行正常，以后若发现有其他bug，再修复。
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target
import treePlottter
import pruning
from sklearn import linear_model
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
class Node(object):
    def __init__(self):
        self.feature_name = None
        self.feature_index = None
        self.subtree = {}
        #噪声
        self.impurity = None
        #离散还是连续
        self.is_continuous = False
        #分割点（连续特征使用
        self.split_value = None
        self.is_leaf = False
        self.leaf_class = None
        self.leaf_num = None
        self.high = -1


class DecisionTree(object):
    '''
    没有针对缺失值的情况作处理。
    '''

    def __init__(self, criterion='gini', pruning=None,data=None):
        '''

        :param criterion: 划分方法选择，'gini', 'infogain', 'gainratio', 三种选项。
        :param pruning:   是否剪枝。 'pre_pruning' 'post_pruning'
        '''
        assert criterion in ('gini', 'infogain', 'gainratio','logistic')
        assert pruning in (None, 'pre_pruning', 'post_pruning')
        self.criterion = criterion
        self.pruning = pruning
        self.data = data

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        '''
        生成决策树
        -------
        :param X:  只支持DataFrame类型数据，因为DataFrame中已有列名，省去一个列名的参数。不支持np.array等其他数据类型
        :param y:
        :return:
        '''

        if self.pruning is not None and (X_val is None or y_val is None):
            raise Exception('you must input X_val and y_val if you are going to pruning')
        #重置索引为0，1，2
        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)

        if X_val is not None:
            X_val.reset_index(inplace=True, drop=True)
            y_val.reset_index(inplace=True, drop=True)
        #获取列名
        self.columns = list(X_train.columns)  # 包括原数据的列名
        self.tree_ = self.generate_tree(X_train, y_train)

        #传入构造好的决策树
        if self.pruning == 'pre_pruning':
            pruning.pre_pruning(X_train, y_train, X_val, y_val, self.tree_)
        elif self.pruning == 'post_pruning':
            pruning.post_pruning(X_train, y_train, X_val, y_val, self.tree_)

        return self

    #y是标签，但是也可以用它来做类别判断，只要在算法执行过程中总是能跟当前特征对应上即可
    def generate_tree(self, X, y):
        my_tree = Node()
        my_tree.leaf_num = 0
        #根据机器学习实战这本书，实际上对于西瓜树上的决策树构造算法的三个基线条件可以总结为两条
        #1、程序遍历完所有划分数据集的属性，即X.empty=true
        #2、每个分支下的所有实例具有相同的分类
        #y.nunique()获取类别数
        if y.nunique() == 1:  # 属于同一类别，那么当前节点就是叶子节点
            my_tree.is_leaf = True
            #标注类别
            my_tree.leaf_class = y.values[0]
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree
        #X.nunique == 1 对行进行去重，如果返回值为1，那么意味着所有特征的特征值相同
        if X.empty or X.nunique == 1:  # 特征为空或者剩下特征的的特征值相同

            my_tree.leaf_class = pd.value_counts(y).index[0]
            my_tree.is_leaf = True

            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree
        best_feature_name=None
        best_impurity=None
        if(self.criterion=='logistic'):
            encoder = OneHotEncoder(sparse_output=False)
            X_encoded = None
            # todo 编码后的数据
            if('密度' in X.columns and '含糖率' in X.columns):
                X_encoded = pd.concat([pd.DataFrame(encoder.fit_transform(X.drop(['密度', '含糖率'], axis=1))), X[['密度','含糖率']]], axis=1)
            elif('密度' in X.columns):
                X_encoded = pd.concat([pd.DataFrame(encoder.fit_transform(X.drop(['密度'], axis=1))), X[['密度']]], axis=1)
            elif('含糖率' in X.columns):
                X_encoded = pd.concat([pd.DataFrame(encoder.fit_transform(X.drop(['含糖率'], axis=1))), X[['含糖率']]], axis=1)
            else:
                X_encoded = pd.DataFrame(encoder.fit_transform(X))
            X_encoded.columns = X_encoded.columns.astype(str)
            # # 获取编码后特征名
            encoded_feature_names = None
            if('密度' in X.columns and '含糖率' in X.columns):
                encoded_feature_names = encoder.get_feature_names_out(X.columns.drop(['密度', '含糖率']))
            elif('密度' in X.columns):
                encoded_feature_names = encoder.get_feature_names_out(X.columns.drop(['密度']))
            elif('含糖率' in X.columns):
                encoded_feature_names = encoder.get_feature_names_out(X.columns.drop(['含糖率']))
            else:
                encoded_feature_names = encoder.get_feature_names_out(X.columns)

            # 利用对率回归计算每个属性每个属性的w，取其中w最高的属性做为划分属性

            best_feature_name, best_impurity = self.choose_best_feature_logistic(X_encoded,X, y,encoded_feature_names)
        else:
            #获取最优属性以及其杂质度（信息增益、增益率、基尼值）
            best_feature_name, best_impurity = self.choose_best_feature_to_split(X, y)

        my_tree.feature_name = best_feature_name
        #杂质度，越低越好
        my_tree.impurity = best_impurity[0]
        #获取最优属性的下标，因为已经将当前节点作为划分节点，所以必要的属性需要进行设置
        my_tree.feature_index = self.columns.index(best_feature_name)
        #获取最优特征（属性）在样本中的所有特征值
        feature_values = X.loc[:, best_feature_name]

        if len(best_impurity) == 1:  # 离散值
            my_tree.is_continuous = False

            unique_vals = pd.unique(self.data.loc[:, best_feature_name])
            #因为当前分支已经利用best_feature进行划分，因此对于当前分支而言，不能再进行划分，因此生成一个将其去掉的子集
            sub_X = X.drop(best_feature_name, axis=1)

            max_high = -1
            for value in unique_vals:
                #对每个特征值对应的分支进行递归构造
                #如果Dv为空,将其标记为叶子节点，其类别为D中样本数最多的类
                if(sub_X[feature_values == value].empty):
                    empty_tree = Node()
                    empty_tree.leaf_num = 0
                    empty_tree.is_leaf = True
                    empty_tree.leaf_class = pd.value_counts(y).index[0]
                    empty_tree.high = 0
                    empty_tree.leaf_num += 1
                    my_tree.subtree[value] = empty_tree
                else:
                    my_tree.subtree[value] = self.generate_tree(sub_X[feature_values == value], y[feature_values == value])
                if my_tree.subtree[value].high > max_high:  # 记录子树下最高的高度
                    max_high = my_tree.subtree[value].high
                my_tree.leaf_num += my_tree.subtree[value].leaf_num

            my_tree.high = max_high + 1

        elif len(best_impurity) == 2:  # 连续值，因为对于连续值而言有分割点
            my_tree.is_continuous = True
            my_tree.split_value = best_impurity[1]
            up_part = '>= {:.3f}'.format(my_tree.split_value)
            down_part = '< {:.3f}'.format(my_tree.split_value)
            #书上提到，对于连续性执行可在分支中重复使用，因此不剔除
            #如果是对数几率回归作为划分准则必须剔除，否则永远都是相同的值，一直在递归
            if(self.criterion=='logistic'):
                sub_X = X.drop(best_feature_name, axis=1)
                if(not sub_X[feature_values >= my_tree.split_value].empty):
                    my_tree.subtree[up_part] = self.generate_tree(sub_X[feature_values >= my_tree.split_value],
                                                                  y[feature_values >= my_tree.split_value])
                else:
                    empty_tree = Node()
                    empty_tree.leaf_num = 0
                    empty_tree.is_leaf = True
                    empty_tree.leaf_class = pd.value_counts(y).index[0]
                    empty_tree.high = 0
                    empty_tree.leaf_num += 1
                    my_tree.subtree[up_part] = empty_tree
                if(not sub_X[feature_values < my_tree.split_value].empty):
                    my_tree.subtree[down_part] = self.generate_tree(sub_X[feature_values < my_tree.split_value],
                                                                    y[feature_values < my_tree.split_value])
                else:
                    empty_tree = Node()
                    empty_tree.leaf_num=0
                    empty_tree.is_leaf = True
                    empty_tree.leaf_class = pd.value_counts(y).index[0]
                    empty_tree.high = 0
                    empty_tree.leaf_num += 1
                    my_tree.subtree[down_part] = empty_tree

            else:
                if (not X[feature_values >= my_tree.split_value].empty):
                    my_tree.subtree[up_part] = self.generate_tree(X[feature_values >= my_tree.split_value],
                                                              y[feature_values >= my_tree.split_value])
                else:
                    empty_tree = Node()
                    empty_tree.leaf_num = 0
                    empty_tree.is_leaf = True
                    empty_tree.leaf_class = pd.value_counts(y).index[0]
                    empty_tree.high = 0
                    empty_tree.leaf_num += 1
                    my_tree.subtree[up_part] = empty_tree
                if (not X[feature_values < my_tree.split_value].empty):
                    my_tree.subtree[down_part] = self.generate_tree(X[feature_values < my_tree.split_value],
                                                                y[feature_values < my_tree.split_value])
                else:
                    empty_tree = Node()
                    empty_tree.leaf_num = 0
                    empty_tree.is_leaf = True
                    empty_tree.leaf_class = pd.value_counts(y).index[0]
                    empty_tree.high = 0
                    empty_tree.leaf_num += 1
                    my_tree.subtree[down_part] = empty_tree
            my_tree.leaf_num += (my_tree.subtree[up_part].leaf_num+my_tree.subtree[down_part].leaf_num)
            my_tree.high = max(my_tree.subtree[up_part].high, my_tree.subtree[down_part].high) + 1

        return my_tree

    def predict(self, X):
        '''
        同样只支持 pd.DataFrame类型数据
        :param X:  pd.DataFrame 类型
        :return:   若
        '''
        if not hasattr(self, "tree_"):
            raise Exception('you have to fit first before predict.')
        if X.ndim == 1:
            return self.predict_single(X)
        else:
            return X.apply(self.predict_single, axis=1)

    def predict_single(self, x, subtree=None):
        '''
        预测单一样本。 实际上这里也可以写成循环，写成递归样本大的时候有栈溢出的风险。
        :param x:
        :param subtree: 根据特征，往下递进的子树。
        :return:
        '''
        if subtree is None:
            subtree = self.tree_

        if subtree.is_leaf:
            return subtree.leaf_class

        if subtree.is_continuous:  # 若是连续值，需要判断是
            if x[subtree.feature_index] >= subtree.split_value:
                return self.predict_single(x, subtree.subtree['>= {:.3f}'.format(subtree.split_value)])
            else:
                return self.predict_single(x, subtree.subtree['< {:.3f}'.format(subtree.split_value)])
        else:
            return self.predict_single(x, subtree.subtree[x[subtree.feature_index]])

    #todo 对率回归
    def choose_best_feature_to_split(self, X, y):
        assert self.criterion in ('gini', 'infogain', 'gainratio','logistic')

        if self.criterion == 'gini':
            return self.choose_best_feature_gini(X, y)
        elif self.criterion == 'infogain':
            return self.choose_best_feature_infogain(X, y)
        elif self.criterion == 'gainratio':
            return self.choose_best_feature_gainratio(X, y)

    def choose_best_feature_logistic(self,X_Encoded,X,y,encoded_feature_names):
        # 遍历每个原始特征
        feature_importance = {}
        features = X.columns
        best_feature_name = None
        best_weight = [float('-inf')]
        feature_weight = [float('-inf')]

        # 使用 OneHotEncoder 对特征进行 One-hot 编码

        lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 使用sklearn的逻辑回归模型
        lr.fit(X_Encoded, y)  # 训练sklearn逻辑回归模型
        count=0
        coef = lr.coef_[0]
        last_two_columns_indices = range(len(X_Encoded.columns) - 2, len(X_Encoded.columns))
        for feature in features:
            #todo 获取最后两列

            if(feature not in {'密度','含糖率'}):
                # 找到One-hot编码后的特征列索引
                indices = [i for i, f in enumerate(encoded_feature_names) if f.startswith(feature)]
                feature_weight = coef[indices].mean()  # 取One-hot编码后权重的平均值

                # 记录每个特征的权重
                feature_importance[feature] = feature_weight
            else:
                #todo 若当前的特征是连续特征，那么应该利用分点方法找出最佳分割点（基于信息增益）
                entD = self.entroy(y)
                m = y.shape[0]
                # 对样本特征值去重，获取训练集中特征所有可能的特征值，两个相同的点显然不能形成区间
                unique_value = pd.unique(X[feature])
                # 连续情形
                unique_value.sort()  # 排序, 用于建立分割点
                # 获取从1到n-1相邻元素的中位点，并返回一个集合
                split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in
                                   range(len(unique_value) - 1)]
                min_ent = float('inf')  # 获取最小信息熵
                min_ent_point = None  # 获取使得信息熵最小的分割点
                # 遍历分割点，从中挑选最小信息熵以及对应分割点
                for split_point_ in split_point_set:

                    # 左半部分
                    Dv1 = y[X[feature] <= split_point_]
                    # 右半部分
                    Dv2 = y[X[feature] > split_point_]
                    # 计算求和内容
                    feature_ent_ = Dv1.shape[0] / m * self.entroy(Dv1) + Dv2.shape[0] / m * self.entroy(Dv2)
                    # 因为ent(D)是确定的，因此我们只需找到最小的求和内容，即可使得Gain(D,a)最大
                    if feature_ent_ < min_ent:
                        min_ent = feature_ent_
                        min_ent_point = split_point_
                # 找到最小求和内容和对应分割点，便得到最大信息增益以及分割点
                gain = entD - min_ent
                feature_weight = coef[last_two_columns_indices[count]]
                feature_importance[feature] = feature_weight
                count+=1
            # 找到权重最大的特征
            if feature_weight > best_weight:
                best_weight = feature_weight
                best_feature_name = feature
        result=None
        if(best_feature_name in ('密度','含糖率')):
           result = [gain, min_ent_point]
        else:
            result = [best_weight]
        return best_feature_name, result
    def choose_best_feature_gini(self, X, y):
        features = X.columns
        best_feature_name = None
        best_gini = [float('inf')]
        #计算每个属性的gini值，找到基尼值最小的属性，并返回属性和基尼值
        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            gini_idex = self.gini_index(X[feature_name], y, is_continuous)
            if gini_idex[0] < best_gini[0]:
                best_feature_name = feature_name
                best_gini = gini_idex

        return best_feature_name, best_gini

    def choose_best_feature_infogain(self, X, y):
        '''
        以返回值中best_info_gain 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益来选择的最佳划分属性，第一个返回值为属性名称，

        '''
        #获取属性列
        features = X.columns
        #初始化最优属性以及信息增益
        best_feature_name = None
        best_info_gain = [float('-inf')]
        #计算总信息熵
        entD = self.entroy(y)
        #遍历每一个属性，分别计算其信息增益，最后返回最大值
        for feature_name in features:
            #判断当前属性是离散属性还是连续属性
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            #计算当前属性的信息增益
            #X[feature_name]当前属性在样本中的所有特征值
            info_gain = self.info_gain(X[feature_name], y, entD, is_continuous)
            if info_gain[0] > best_info_gain[0]:
                best_feature_name = feature_name
                best_info_gain = info_gain

        return best_feature_name, best_info_gain

    def choose_best_feature_gainratio(self, X, y):
        '''
        以返回值中best_gain_ratio 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益率来选择的最佳划分属性，第一个返回值为属性名称，第二个为最佳划分属性对应的信息增益率
        '''
        features = X.columns
        best_feature_name = None
        best_gain_ratio = [float('-inf')]
        entD = self.entroy(y)

        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            info_gain_ratio = self.info_gainRatio(X[feature_name], y, entD, is_continuous)
            if info_gain_ratio[0] > best_gain_ratio[0]:
                best_feature_name = feature_name
                best_gain_ratio = info_gain_ratio

        return best_feature_name, best_gain_ratio

    def gini_index(self, feature, y, is_continuous=False):
        '''
        计算基尼指数， 对于连续值，选择基尼系统最小的点，作为分割点
        -------
        :param feature:
        :param y:
        :return:
        '''
        #获取传入集合的的样本数
        m = y.shape[0]
        #去重
        unique_value = pd.unique(feature)
        if is_continuous:
            unique_value.sort()  # 排序, 用于建立分割点
            # 这里其实也可以直接用feature值作为分割点，但这样会出现空集， 所以还是按照书中4.7式建立分割点。好处是不会出现空集
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]

            min_gini = float('inf')
            min_gini_point = None
            for split_point_ in split_point_set:  # 遍历所有的分割点，寻找基尼指数最小的分割点
                Dv1 = y[feature <= split_point_]
                Dv2 = y[feature > split_point_]
                gini_index = Dv1.shape[0] / m * self.gini(Dv1) + Dv2.shape[0] / m * self.gini(Dv2)

                if gini_index < min_gini:
                    min_gini = gini_index
                    min_gini_point = split_point_
            return [min_gini, min_gini_point]
        else:
            gini_index = 0
            for value in unique_value:
                Dv = y[feature == value]
                m_dv = Dv.shape[0]
                gini = self.gini(Dv)  # 原书4.5式
                gini_index += m_dv / m * gini  # 4.6式

            return [gini_index]

    def gini(self, y):
        p = pd.value_counts(y) / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return gini

    def info_gain(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益
        ------
        :param feature: 当前特征下所有样本值
        :param y:       对应标签值
        :return:        当前特征的信息增益, list类型，若当前特征为离散值则只有一个元素为信息增益，若为连续值，则第一个元素为信息增益，第二个元素为切分点
        '''
        #获取样本总数|D|
        m = y.shape[0]
        #对样本特征值去重，获取训练集中特征所有可能的特征值
        unique_value = pd.unique(feature)
        #连续情形
        if is_continuous:
            unique_value.sort()  # 排序, 用于建立分割点
            #获取从1到n-1相邻元素的中位点，并返回一个集合
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]
            min_ent = float('inf')  # 获取最小信息熵
            min_ent_point = None # 获取使得信息熵最小的分割点
            #遍历分割点，从中挑选最小信息熵以及对应分割点
            for split_point_ in split_point_set:

                #左半部分
                Dv1 = y[feature <= split_point_]
                #右半部分
                Dv2 = y[feature > split_point_]
                #计算求和内容
                feature_ent_ = Dv1.shape[0] / m * self.entroy(Dv1) + Dv2.shape[0] / m * self.entroy(Dv2)
                #因为ent(D)是确定的，因此我们只需找到最小的求和内容，即可使得Gain(D,a)最大
                if feature_ent_ < min_ent:
                    min_ent = feature_ent_
                    min_ent_point = split_point_
            #找到最小求和内容和对应分割点，便得到最大信息增益以及分割点
            gain = entD - min_ent

            return [gain, min_ent_point]

        else:
            #初始化特征的信息熵
            feature_ent = 0
            for value in unique_value:
                Dv = y[feature == value]  # 当前特征中取值为 value 的样本，即书中的 D^{v}
                feature_ent += Dv.shape[0] / m * self.entroy(Dv)

            gain = entD - feature_ent  # 原书中4.2式
            return [gain]

    def info_gainRatio(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益率 参数和info_gain方法中参数一致
        ------
        :param feature:
        :param y:
        :param entD:
        :return:
        '''

        if is_continuous:
            # 对于连续值，以最大化信息增益选择划分点之后，计算信息增益率，注意，在选择划分点之后，需要对信息增益进行修正，要减去log_2(N-1)/|D|，N是当前特征的取值个数，D是总数据量。
            # 修正原因是因为：当离散属性和连续属性并存时，C4.5算法倾向于选择连续特征做最佳树分裂点
            # 信息增益修正中，N的值，网上有些资料认为是“可能分裂点的个数”，也有的是“当前特征的取值个数”，这里采用“当前特征的取值个数”。
            # 这样 (N-1)的值，就是去重后的“分裂点的个数” , 即在info_gain函数中，split_point_set的长度，个人感觉这样更加合理。有时间再看看原论文吧。

            gain, split_point = self.info_gain(feature, y, entD, is_continuous)
            p1 = np.sum(feature <= split_point) / feature.shape[0]  # 小于或划分点的样本占比
            p2 = 1 - p1  # 大于划分点样本占比
            IV = -(p1 * np.log2(p1) + p2 * np.log2(p2))

            grain_ratio = (gain - np.log2(feature.nunique()) / len(y)) / IV  # 对信息增益修正
            return [grain_ratio, split_point]
        else:
            p = pd.value_counts(feature) / feature.shape[0]  # 当前特征下 各取值样本所占比率
            IV = np.sum(-p * np.log2(p))  # 原书4.4式
            grain_ratio = self.info_gain(feature, y, entD, is_continuous)[0] / IV
            return [grain_ratio]

    def entroy(self, y):
        #pd.value_counts(y)统计出各类样本个数，y.shape[0]为样本总数
        #因此p为各类样本所占比率，注意p是一个向量
        p = pd.value_counts(y) / y.shape[0]  # 计算各类样本所占比率
        ent = np.sum(-p * np.log2(p))
        return ent


if __name__ == '__main__':

    # 4.3
    data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch4--决策树\watermelon3_0_Ch.csv'
    data3 = pd.read_csv(data_path, index_col=0)

    tree = DecisionTree('infogain', None,data3)
    #第一个参数是特征，第二个参数是标签
    tree.fit(data3.iloc[:, :8], data3.iloc[:, 8])
    treePlottter.create_plot(tree.tree_)

    # 4.4
    # data_path2 = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch4--决策树\watermelon2_0_Ch.txt'
    # data = pd.read_table(data_path2, encoding='utf8', delimiter=',', index_col=0)
    # # data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch4--决策树\watermelon3_0_Ch.csv'
    # # data = pd.read_csv(data_path, index_col=0)
    #
    # train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    # #转换成数组下标
    # train = [i - 1 for i in train]
    # #分别获取训练集数据和标签
    # X = data.iloc[train, :6]
    # y = data.iloc[train, 6]
    #
    # test = [4, 5, 8, 9, 11, 12, 13]
    # test = [i - 1 for i in test]
    #
    # X_val = data.iloc[test, :6]
    # y_val = data.iloc[test, 6]
    #
    # tree = DecisionTree('gini', None,data)
    # tree.fit(X, y, X_val, y_val)
    # print(np.mean(tree.predict(X_val) == y_val))
    # treePlottter.create_plot(tree.tree_)

    # 4.5
    data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch4--决策树\watermelon3_0_Ch.csv'
    data = pd.read_csv(data_path, index_col=0)

    train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    #转换成数组下标
    train = [i - 1 for i in train]
    #分别获取训练集数据和标签
    X = data.iloc[train, :8]
    y = data.iloc[train, 8]

    test = [4, 5, 8, 9, 11, 12, 13]
    test = [i - 1 for i in test]

    X_val = data.iloc[test, :8]
    y_val = data.iloc[test, 8]

    tree = DecisionTree('logistic', None,data)
    tree.fit(X, y, X_val, y_val)
    print(np.mean(tree.predict(X_val) == y_val))
    treePlottter.create_plot(tree.tree_)
