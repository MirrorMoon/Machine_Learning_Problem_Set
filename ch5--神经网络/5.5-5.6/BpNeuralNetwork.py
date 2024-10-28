import numpy as np
import copy
import pandas as pd
import bpnnUtil
from sklearn import datasets


class BpNN(object):
    def __init__(self, layer_dims_, learning_rate=0.1, seed=16, initializer='he', optimizer='gd'):
        """
               初始化BP神经网络模型

               :param layer_dims_: 每一层的节点数
               :param learning_rate: 学习率
               :param seed: 随机种子，确保结果一致性
               :param initializer: 权重初始化方法， 'he' 或 'xavier'
               :param optimizer: 优化器选择，可选 'gd'、 'sgd'、'adam'、'momentum'
        """
        #layer_dims_ 表示神经网络中每一层的维度。它是一个列表，其中每个元素指定对应层中的神经元数量。例如，如果 layer_dims_ 为 [3, 5, 2]，这意味着神经网络有三层：第一层有 3 个神经元，第二层有 5 个神经元，第三层有 2 个神经元。
        self.layer_dims_ = layer_dims_
        self.learning_rate = learning_rate
        self.seed = seed
        self.initializer = initializer
        self.optimizer = optimizer

    def fit(self, X_, y_, num_epochs=100):
        """
               模型训练函数

               :param X_: 输入数据
               :param y_: 输出标签
               :param num_epochs: 训练次数
               """
        # 获取特征的行数和列数
        m, n = X_.shape
        #这里利用深拷贝创建一个原对象的副本
        layer_dims_ = copy.deepcopy(self.layer_dims_)
        #`layer_dims_.insert(0, n)` 这行代码的作用是将输入层的维度（即特征数量 `n`）插入到 `layer_dims_` 列表的开头。
        # 这样做是为了确保神经网络的输入层维度正确地包含在网络的层次结构中。
        layer_dims_.insert(0, n)

        # 如果标签是一维的，将其转换为二维，为了保证兼容性[1,2,3],[[1,2,3]]
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        assert self.initializer in ('he', 'xavier')
        #初始化神经网络的参数，这里的参数包括权重和偏置，初始化方法有两种，一种是he初始化，一种是xavier初始化
        if self.initializer == 'he':
            self.parameters_ = bpnnUtil.he_initializer(layer_dims_, self.seed)
        elif self.initializer == 'xavier':
            self.parameters_ = bpnnUtil.xavier_initializer(layer_dims_, self.seed)

        assert self.optimizer in ('gd', 'sgd', 'adam', 'momentum')
        if self.optimizer == 'gd':
            parameters_, costs = self.optimizer_gd(X_, y_, self.parameters_, num_epochs, self.learning_rate)
        elif self.optimizer == 'sgd':
            parameters_, costs = self.optimizer_sgd(X_, y_, self.parameters_, num_epochs, self.learning_rate, self.seed)
        elif self.optimizer == 'momentum':
            parameters_, costs = self.optimizer_sgd_monment(X_, y_, self.parameters_, beta=0.9, num_epochs=num_epochs,
                                                            learning_rate=self.learning_rate, seed=self.seed)
        elif self.optimizer == 'adam':
            parameters_, costs = self.optimizer_sgd_adam(X_, y_, self.parameters_, beta1=0.9, beta2=0.999, epsilon=1e-7,
                                                         num_epochs=num_epochs, learning_rate=self.learning_rate,
                                                         seed=self.seed)

        self.parameters_ = parameters_
        self.costs = costs

        return self

    def predict(self, X_):
        if not hasattr(self, "parameters_"):
            raise Exception('you have to fit first before predict.')

        a_last, _ = self.forward_L_layer(X_, self.parameters_)
        if a_last.shape[1] == 1:
            predict_ = np.zeros(a_last.shape)
            predict_[a_last>=0.5] = 1
        else:
            predict_ = np.argmax(a_last, axis=1)
        return predict_

    def compute_cost(self, y_hat_, y_):
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)
        if y_.shape[1] == 1:
            cost = bpnnUtil.MSE(y_hat_, y_)
        else:
            cost = bpnnUtil.cross_entry_softmax(y_hat_, y_)
        return cost


    def backward_one_layer(self, da_, cache_, activation_):
        # 在activation_ 为'softmax'时， da_实际上输入是y_， 并不是输出层的梯度
        (a_pre_, w_, b_, z_) = cache_  # 从缓存中获取前一层的激活值，当前层的权重，偏置以及神经元的输入

        m = da_.shape[0]  # 获取样本数量

        assert activation_ in ('sigmoid', 'relu', 'softmax')  # 确保激活函数是'sigmoid'、'relu'或'softmax'

        # 根据激活函数类型计算dz_
        if activation_ == 'sigmoid':
            dz_ = bpnnUtil.sigmoid_backward(da_, z_)
        elif activation_ == 'relu':
            dz_ = bpnnUtil.relu_backward(da_, z_)
        else:
            dz_ = bpnnUtil.softmax_backward(da_, z_)

        # 计算权重和偏置的梯度
        #书5.11式，5.12式,5.15式
        #dz是gj,a_pre_是bh,m是均方误差的样本数m
        #注意这里使用大量的向量运算是因为我们不像书上只计算某一个神经元的偏导，而是将一层内所有神经元的梯度都计算出来，而所有神经元的梯度的计算方法显然一致，因此引入向量计算
        dw = np.dot(dz_.T, a_pre_) / m
        db = np.sum(dz_, axis=0, keepdims=True) / m
        #
        # da_pre =np.dot(np.dot(dz_, w_),np.dot(a_pre_.T,a_pre_))  # 计算前一层神经元的梯度，这是书上的5.15式，但是结果是错误的
        #这里是常规的反向传播算法，即计算出损失函数相对于前一层的激活值的梯度
        #然后这个da_pre_在下次循环的时候就是上一层的da_，这样就可以一层一层的反向传播计算梯度
        #这里使用点积的形式可以完美的处理gj*whj以及针对这个结果的加和
        #比如考虑17x3x3这样的一个三层网络，那么dz_的维度是17x3，w_的维度是3x3，那么点积的结果是17x3，这个结果就是gj*whj
        da_pre = np.dot(dz_, w_)  # 计算前一层的激活值的梯度

        # 确保梯度的形状与相应的参数一致
        assert dw.shape == w_.shape
        assert db.shape == b_.shape
        assert da_pre.shape == a_pre_.shape

        return da_pre, dw, db  # 返回前一层的激活值梯度以及链接权重梯度和当前层偏置梯度



    def backward_L_layer(self, a_last, y_, caches):
        # 初始化梯度字典
        grads = {}
        L = len(caches)  # 获取层数（不包含输入层

        # 如果标签是一维的，将其转换为二维即[[]]，为了保证兼容性
        if y_.ndim == 1:
            y_ = y_.reshape(-1, 1)

        # 如果标签只有一列，表示二分类
        #todo 均方误差的梯度计算
        if y_.shape[1] == 1:
            # 计算输出层的梯度
            #将最后一层即最终的输出层的输入作为反向传播的输入，计算输出层的梯度
            #计算出输出层的梯度，即损失函数E对输出层y^的偏导数，这里的损失函数是交叉熵
            # da_last = -(y_ / a_last - (1 - y_) / (1 - a_last))
            #使用均方误差
            da_last = a_last - y_
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(da_last, caches[L - 1], 'sigmoid')
            #todo softmax的梯度计算
        else:  # 多分类
            # 计算softmax的梯度
            da_pre_L_1, dwL_, dbL_ = self.backward_one_layer(y_, caches[L - 1], 'softmax')

        # 保存损失函数e对上一层（隐层）的激活值的梯度
        grads['da' + str(L)] = da_pre_L_1
        #当前层的权重梯度
        grads['dW' + str(L)] = dwL_
        #当前层的偏置梯度
        grads['db' + str(L)] = dbL_

        # 反向传播每一层
        for i in range(L - 1, 0, -1):
            da_pre_, dw, db = self.backward_one_layer(grads['da' + str(i + 1)], caches[i - 1], 'relu')

            # 保存每一层的梯度
            grads['da' + str(i)] = da_pre_
            grads['dW' + str(i)] = dw
            grads['db' + str(i)] = db

        return grads

    def forward_one_layer(self, a_pre_, w_, b_, activation_):
        # 计算线性部分 z = a_pre_ * w_.T + b_
        z_ = np.dot(a_pre_, w_.T) + b_
        z_ = np.array(z_, dtype=np.float64)  # 强制转换为float64类型

        # 确保激活函数是 'sigmoid'、'relu' 或 'softmax'
        assert activation_ in ('sigmoid', 'relu', 'softmax')

        # 根据激活函数类型计算当前层激活值 a_
        if activation_ == 'sigmoid':
            a_ = bpnnUtil.sigmoid(z_)
        elif activation_ == 'relu':
            a_ = bpnnUtil.relu(z_)
        else:
            a_ = bpnnUtil.softmax(z_)

        # 将前向传播过程中产生的数据保存下来，在反向传播传播过程计算梯度的时候要用上
        #因为对于反向传播而言，需要知道当前层的输入，权重，偏置，输出，以及上一层的输出，所以这里将这些数据保存下来
        cache_ = (a_pre_, w_, b_, z_)
        return a_, cache_

    def forward_L_layer(self, X_, parameters_):
        # 获取神经网络的层数（除了输入层
        L_ = int(len(parameters_) / 2)
        caches = []
        # 初始化输入层
        a_ = X_

        # 前向传播每一层（除了最后一层）
        for i in range(1, L_):
            w_ = parameters_['W' + str(i)]
            b_ = parameters_['b' + str(i)]
            #新建一个输入层的临时变量，用于保存上一层的输入
            a_pre_ = a_
            #计算出当前层的输出和缓存，缓存用于反向传播，将当前层的输出继续作为下一层的输入
            a_, cache_ = self.forward_one_layer(a_pre_, w_, b_, 'sigmoid')
            caches.append(cache_)

        # 前向传播最后一层
        w_last = parameters_['W' + str(L_)]
        b_last = parameters_['b' + str(L_)]

        # 根据输出层的维度选择激活函数，sigmoid适用于二分类，softmax适用于多分类
        #若输出层的维度为1，则意味着只有一个神经元，所以是二分类问题，此时激活函数为sigmoid
        #若输出层的维度大于1，则意味着有多个神经元，所以是多分类问题，此时激活函数为softmax
        if w_last.shape[0] == 1:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'sigmoid')
        else:
            a_last, cache_ = self.forward_one_layer(a_, w_last, b_last, 'softmax')
        #将最后一层的缓存加入列表，用于反向传播时的梯度计算
        caches.append(cache_)
        return a_last, caches

#西瓜书上说学习率一般设置为0.1，这里默认设置为0.1
    def optimizer_gd(self, X_, y_, parameters_, num_epochs, learning_rate):
        # 初始化一个空列表，用于存储每次迭代的损失值
        costs = []
        for i in range(num_epochs):
            # 前向传播，计算当前参数下的输出
            a_last, caches = self.forward_L_layer(X_, parameters_)
            # 反向传播，计算梯度
            grads = self.backward_L_layer(a_last, y_, caches)
            # 利用计算出来的梯度更新参数
            parameters_ = bpnnUtil.update_parameters_with_gd(parameters_, grads, learning_rate)
            # 计算当前epoch的损失值
            cost = self.compute_cost(a_last, y_)
            # 将损失值添加到costs列表中
            costs.append(cost)
        # 返回更新后的参数和损失值列表
        return parameters_, costs


    def optimizer_sgd(self, X_, y_, parameters_, num_epochs, learning_rate, seed):
        '''
        随机梯度下降（SGD）优化器。
        在SGD中，参数更新步骤与GD相似，但梯度是使用单个样本计算的。

        :param X_: 输入数据
        :param y_: 输出标签
        :param parameters_: 初始化的参数
        :param num_epochs: 训练次数
        :param learning_rate: 学习率
        :param seed: 随机种子，确保结果一致性
        :return: 更新后的参数和损失值历史
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]

        for _ in range(num_epochs):
            # 随机选择一个样本
            random_index = np.random.randint(0, m_)

            # 选定样本的前向传播
            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)

            # 反向传播计算梯度
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            # 使用计算出的梯度更新参数
            parameters_ = bpnnUtil.update_parameters_with_sgd(parameters_, grads, learning_rate)

            # 计算整个数据集的损失值
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)

            # 存储损失值
            costs.append(cost)

        return parameters_, costs

    # def optimizer_sgd(self, X_, y_, parameters_, num_epochs, learning_rate, seed):
    #     '''
    #     sgd中，更新参数步骤和gd是一致的，只不过在计算梯度的时候是用一个样本而已。
    #     '''
    #     np.random.seed(seed)
    #     costs = []
    #     m_ = X_.shape[0]
    #     for _ in range(num_epochs):
    #         random_index = np.random.randint(0, m_)
    #
    #         a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
    #         grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)
    #
    #         parameters_ = bpnnUtil.update_parameters_with_sgd(parameters_, grads, learning_rate)
    #
    #         a_last_cost, _ = self.forward_L_layer(X_, parameters_)
    #
    #         cost = self.compute_cost(a_last_cost, y_)
    #
    #         costs.append(cost)
    #
    #     return parameters_, costs

    def optimizer_sgd_monment(self, X_, y_, parameters_, beta, num_epochs, learning_rate, seed):
        '''

        :param X_:
        :param y_:
        :param parameters_: 初始化的参数
        :param v_:          梯度的指数加权移动平均数
        :param beta:        冲量大小，
        :param num_epochs:
        :param learning_rate:
        :param seed:
        :return:
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        velcoity = bpnnUtil.initialize_velcoity(parameters_)
        for _ in range(num_epochs):
            random_index = np.random.randint(0, m_)

            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            parameters_, v_ = bpnnUtil.update_parameters_with_sgd_momentum(parameters_, grads, velcoity, beta,
                                                                           learning_rate)
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs



    def optimizer_sgd_adam(self, X_, y_, parameters_, beta1, beta2, epsilon, num_epochs, learning_rate, seed):
        '''
        使用Adam优化器进行参数更新。

        :param X_: 输入数据
        :param y_: 输出标签
        :param parameters_: 初始化的参数
        :param beta1: 一阶矩估计的指数衰减率
        :param beta2: 二阶矩估计的指数衰减率
        :param epsilon: 防止除零操作的小数
        :param num_epochs: 训练次数
        :param learning_rate: 学习率
        :param seed: 随机种子，确保结果一致性
        :return: 更新后的参数和损失值历史
        '''
        np.random.seed(seed)
        costs = []
        m_ = X_.shape[0]
        # 初始化一阶矩估计和二阶矩估计（也就是动量和RMSprop）
        velcoity, square_grad = bpnnUtil.initialize_adam(parameters_)
        for epoch in range(num_epochs):
            # 随机选择一个样本
            random_index = np.random.randint(0, m_)

            # 前向传播，计算当前参数下的输出
            a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
            # 反向传播，计算梯度
            grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)

            # 使用Adam优化器更新参数
            parameters_, velcoity, square_grad = bpnnUtil.update_parameters_with_sgd_adam(parameters_, grads, velcoity,
                                                                                          square_grad, epoch + 1,
                                                                                          learning_rate, beta1, beta2,
                                                                                          epsilon)
            # 计算整个数据集的损失值
            a_last_cost, _ = self.forward_L_layer(X_, parameters_)
            cost = self.compute_cost(a_last_cost, y_)
            costs.append(cost)

        return parameters_, costs
    # def optimizer_sgd_adam(self, X_, y_, parameters_, beta1, beta2, epsilon, num_epochs, learning_rate, seed):
    #     '''
    #
    #     :param X_:
    #     :param y_:
    #     :param parameters_: 初始化的参数
    #     :param v_:          梯度的指数加权移动平均数
    #     :param beta:        冲量大小，
    #     :param num_epochs:
    #     :param learning_rate:
    #     :param seed:
    #     :return:
    #     '''
    #     np.random.seed(seed)
    #     costs = []
    #     m_ = X_.shape[0]
    #     velcoity, square_grad = bpnnUtil.initialize_adam(parameters_)
    #     for epoch in range(num_epochs):
    #         random_index = np.random.randint(0, m_)
    #
    #         a_last, caches = self.forward_L_layer(X_[[random_index], :], parameters_)
    #         grads = self.backward_L_layer(a_last, y_[[random_index], :], caches)
    #
    #         parameters_, velcoity, square_grad = bpnnUtil.update_parameters_with_sgd_adam(parameters_, grads, velcoity,
    #                                                                                       square_grad, epoch + 1,
    #                                                                                       learning_rate, beta1, beta2,
    #                                                                                       epsilon)
    #         a_last_cost, _ = self.forward_L_layer(X_, parameters_)
    #         cost = self.compute_cost(a_last_cost, y_)
    #         costs.append(cost)
    #
    #     return parameters_, costs


if __name__ == '__main__':
    # 5.5

    data_path = r'D:\Machine_Learning_Problem_Set\data\watermelon3_0_Ch.csv'
    data3 = pd.read_csv(data_path, index_col=0)
    #对非数值离散属性进行one hot编码将其转为连续数值属性，因为神经网络的输入、激活函数以及输出都是数值型数据
    data = pd.get_dummies(data3, columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    #将标签转为0、1
    data['好瓜'] = data['好瓜'].map({'是': 1, '否': 0})
    #获取特征
    X_test = data.drop('好瓜', axis=1)
    #获取标签
    y_test = data['好瓜']
    #累计bp
    bp = BpNN([3, 1], learning_rate=0.1, optimizer='gd')
    bp.fit(X_test.values, y_test.values, num_epochs=200)
    #标准bp
    bp1 = BpNN([3, 1], learning_rate=0.1, optimizer='sgd')
    bp1.fit(X_test.values, y_test.values, num_epochs=200)

    bpnnUtil.plot_costs([bp.costs, bp1.costs], ['gd_cost', 'sgd_cost'])

    # 5.6
    iris = datasets.load_iris()
    # X = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    # X = (X - np.mean(X, axis=0)) / np.var(X, axis=0)
    #
    # y = pd.Series(iris['target_names'][iris['target']])
    # y = pd.get_dummies(y)
    #
    # bp = BpNN([3, 3], learning_rate=0.003, optimizer='adam')
    # bp.fit(X.values, y.values, num_epochs=2000)
    #
    # bp1 = BpNN([3, 3], learning_rate=0.003, optimizer='sgd')
    # bp1.fit(X.values, y.values, num_epochs=2000)
    #
    # bpnnUtil.plot_costs([bp.costs, bp1.costs], ['adam_cost', 'sgd_cost'])
