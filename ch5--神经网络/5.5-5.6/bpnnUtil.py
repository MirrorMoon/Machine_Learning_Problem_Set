import numpy as np
from matplotlib import pyplot as plt


def xavier_initializer(layer_dims_, seed=16):
    """
     Xavier初始化方法，适用于sigmoid/tanh激活函数。
     :param layer_dims_: 神经网络每层的神经元数量。
     :param seed: 随机种子，保证每次运行结果相同。
     :return: 初始化后的权重和偏置字典。
     """
    # 设置随机种子，以确保每次运行代码时生成的随机数相同
    np.random.seed(seed)

    # 初始化一个空字典，用于存储权重和偏置
    parameters_ = {}
    # 获取神经网络的层数
    num_L = len(layer_dims_)
    # 遍历每一层（除了最后一层）
    for l in range(num_L - 1):
        # 使用 Xavier 初始化方法生成权重矩阵
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(1 / layer_dims_[l])
        # 初始化输出层（可以是隐层，也可以是最终的输出层）偏置为零，所以肯定从l+1开始
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        # 将生成的每一层（两两相连）权重矩阵存储在字典中
        parameters_['W' + str(l + 1)] = temp_w
        # 将生成的每一层（两两相连）偏置向量存储在字典中
        parameters_['b' + str(l + 1)] = temp_b


    return parameters_


def he_initializer(layer_dims_, seed=16):
    np.random.seed(seed)

    parameters_ = {}
    num_L = len(layer_dims_)
    for l in range(num_L - 1):
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(2 / layer_dims_[l])
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        parameters_['W' + str(l + 1)] = temp_w
        parameters_['b' + str(l + 1)] = temp_b

    return parameters_


def cross_entry_sigmoid(y_hat_, y_):
    '''
    计算在二分类时的交叉熵
    :param y_hat_:  模型输出值
    :param y_:      样本真实标签值
    :return:
    '''

    m = y_.shape[0]
    loss = -(np.dot(y_.T, np.log(y_hat_)) + np.dot(1 - y_.T, np.log(1 - y_hat_))) / m

    return np.squeeze(loss)


def cross_entry_softmax(y_hat_, y_):
    '''
    计算多分类时的交叉熵
    :param y_hat_:
    :param y_:
    :return:
    '''
    m = y_.shape[0]
    loss = -np.sum(y_ * np.log(y_hat_)) / m
    return loss


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    z -= np.max(z)  # 防止过大，超出限制，导致计算结果为 nan
    z_exp = np.exp(z)
    softmax_z = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return softmax_z


def sigmoid_backward(da_, cache_z):
    a = 1 / (1 + np.exp(-cache_z))
    #da是损失函数对激活值y^的偏导，a*(1-a)是激活函数对z的偏导,也就是书上的5.10式，只不过损失函数换成了交叉熵，但是逻辑是一样的。da总是表示来自下一层损失函数对激活值的偏导
    dz_ = da_ * a * (1 - a)
    #由于梯度 dz_ 是从 z 上反向传播计算出来的，因此 dz_ 和 z 必须有相同的形状。这是因为每一个元素 z 对应的梯度 dz_ 也应该是一个相同维度的标量。
    assert dz_.shape == cache_z.shape
    return dz_


def softmax_backward(y_, cache_z):
    #
    a = softmax(cache_z)
    dz_ = a - y_
    assert dz_.shape == cache_z.shape
    return dz_


def relu_backward(da_, cache_z):
    dz = np.array(da_, copy=True)
    dz[cache_z <= 0] = 0
    assert (dz.shape == cache_z.shape)

    return dz


def update_parameters_with_gd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)
    #转换成float类型，否则会报错
    transDictObject2Float(parameters_)
    transDictObject2Float(grads)

    #L+1是因为向下取整后L_=2,那么range(1,2)只有1，所以要加1遍历最后一层
    #(1,L_)w1 = w1-leraning_rate*dw1
    for l in range(1, L_ + 1):
        #利用计算出来的梯度逐层开始更新参数
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def update_parameters_with_sgd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)
    transDictObject2Float(parameters_)
    transDictObject2Float(grads)
    for l in range(1, L_ + 1):
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def initialize_velcoity(paramters):
    v = {}

    L_ = int(len(paramters) / 2)

    for l in range(1, L_ + 1):
        v['dW' + str(l)] = np.zeros(paramters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(paramters['b' + str(l)].shape)

    return v


def update_parameters_with_sgd_momentum(parameters, grads, velcoity, beta, learning_rate):
    L_ = int(len(parameters) / 2)

    for l in range(1, L_ + 1):
        velcoity['dW' + str(l)] = beta * velcoity['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        velcoity['db' + str(l)] = beta * velcoity['db' + str(l)] + (1 - beta) * grads['db' + str(l)]

        parameters['W' + str(l)] -= learning_rate * velcoity['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * velcoity['db' + str(l)]

    return parameters, velcoity



def initialize_adam(paramters_):
    # 计算神经网络的层数
    l = int(len(paramters_) / 2)
    # 初始化字典，用于存储梯度的平方和动量
    #梯度的平方2阶矩会使用，动量1阶矩使用
    square_grad = {}
    velcoity = {}
    # 遍历每一层
    for i in range(l):
        # 初始化梯度的平方和动量为零矩阵
        square_grad['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
        square_grad['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
        velcoity['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
        velcoity['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
    # 返回初始化后的字典
    return velcoity, square_grad

# def initialize_adam(paramters_):
#     l = int(len(paramters_) / 2)
#     square_grad = {}
#     velcoity = {}
#     for i in range(l):
#
#         for i in range(l):
#             square_grad['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
#             square_grad['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
#             velcoity['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
#             velcoity['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
#         return velcoity, square_grad



def update_parameters_with_sgd_adam(parameters_, grads_, velcoity, square_grad, epoch, learning_rate=0.1, beta1=0.9,
                                    beta2=0.999, epsilon=1e-8):
    # 计算神经网络的层数
    l = int(len(parameters_) / 2)

    # 遍历每一层
    for i in range(l):
        # 更新动量
        # 注意，动量的状态是根据主算法迭代次数来更新的，所以代码的后半部分用的是velcoity['dW' + str(i + 1)]
        velcoity['dW' + str(i + 1)] = beta1 * velcoity['dW' + str(i + 1)] + (1 - beta1) * grads_['dW' + str(i + 1)]
        velcoity['db' + str(i + 1)] = beta1 * velcoity['db' + str(i + 1)] + (1 - beta1) * grads_['db' + str(i + 1)]

        # 对动量进行偏差修正，为了让动量特性在最开始就能起作用，显然当迭代次数t足够大时，分母就会趋近于1，即mt^无限趋于mt
        vw_correct = velcoity['dW' + str(i + 1)] / (1 - np.power(beta1, epoch))
        vb_correct = velcoity['db' + str(i + 1)] / (1 - np.power(beta1, epoch))

        # 更新梯度的平方和
        square_grad['dW' + str(i + 1)] = beta2 * square_grad['dW' + str(i + 1)] + (1 - beta2) * (grads_['dW' + str(i + 1)] ** 2)
        square_grad['db' + str(i + 1)] = beta2 * square_grad['db' + str(i + 1)] + (1 - beta2) * (grads_['db' + str(i + 1)] ** 2)

        # 对梯度的平方和进行偏差修正
        sw_correct = square_grad['dW' + str(i + 1)] / (1 - np.power(beta2, epoch))
        sb_correct = square_grad['db' + str(i + 1)] / (1 - np.power(beta2, epoch))

        # 更新参数
        parameters_['W' + str(i + 1)] -= learning_rate * vw_correct / np.sqrt(sw_correct + epsilon)
        parameters_['b' + str(i + 1)] -= learning_rate * vb_correct / np.sqrt(sb_correct + epsilon)

    return parameters_, velcoity, square_grad

# def update_parameters_with_sgd_adam(parameters_, grads_, velcoity, square_grad, epoch, learning_rate=0.1, beta1=0.9,
#                                     beta2=0.999, epsilon=1e-8):
#     l = int(len(parameters_) / 2)
#
#     for i in range(l):
#         velcoity['dW' + str(i + 1)] = beta1 * velcoity['dW' + str(i + 1)] + (1 - beta1) * grads_['dW' + str(i + 1)]
#         velcoity['db' + str(i + 1)] = beta1 * velcoity['db' + str(i + 1)] + (1 - beta1) * grads_['db' + str(i + 1)]
#
#         vw_correct = velcoity['dW' + str(i + 1)] / (1 - np.power(beta1, epoch))         # 这里是对迭代初期的梯度进行修正
#         vb_correct = velcoity['db' + str(i + 1)] / (1 - np.power(beta1, epoch))
#
#         square_grad['dW' + str(i + 1)] = beta2 * square_grad['dW' + str(i + 1)] + (1 - beta2) * (
#                     grads_['dW' + str(i + 1)] ** 2)
#         square_grad['db' + str(i + 1)] = beta2 * square_grad['db' + str(i + 1)] + (1 - beta2) * (
#                     grads_['db' + str(i + 1)] ** 2)
#
#         sw_correct = square_grad['dW' + str(i + 1)] / (1 - np.power(beta2, epoch))
#         sb_correct = square_grad['db' + str(i + 1)] / (1 - np.power(beta2, epoch))
#
#         parameters_['W' + str(i + 1)] -= learning_rate * vw_correct / np.sqrt(sw_correct + epsilon)
#         parameters_['b' + str(i + 1)] -= learning_rate * vb_correct / np.sqrt(sb_correct + epsilon)
#
#     return parameters_, velcoity, square_grad


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plot_costs(costs, labels, colors=None):
    if colors is None:
        colors = ['blue', 'lightcoral']

    ax = plt.subplot()
    assert len(costs) == len(labels)
    for i in range(len(costs)):
        ax.plot(costs[i], color=colors[i], label=labels[i])
    set_ax_gray(ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('num epochs')
    ax.set_ylabel('cost')
    plt.show()
def transDictObject2Float(dict_):
    for key, value in dict_.items():
        if isinstance(value, np.ndarray) and value.dtype == object:
            # 尝试将数组转换为 float 类型
            try:
                dict_[key] = value.astype(np.float64)
            except ValueError as e:
                print(f"无法将 {key} 转换为 float: {e}")