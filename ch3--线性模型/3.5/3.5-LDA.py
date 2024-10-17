import numpy as np  # 导入NumPy库，用于数组和数学运算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from matplotlib import pyplot as plt  # 导入Matplotlib库，用于绘图


class LDA(object):  # 定义一个LDA类，用于线性判别分析

    def fit(self, X_, y_, plot_=False):  # 定义拟合函数，接受特征X_、标签y_和是否绘图的参数plot_
        pos = y_ == 1  # 找到正类样本
        neg = y_ == 0  # 找到负类样本
        X0 = X_[neg]  # 负类样本特征
        X1 = X_[pos]  # 正类样本特征
        #keepdims=True是保证原始数组（矩阵）的维度
        u0 = X0.mean(0, keepdims=True)  # 计算负类样本的均值 (1, n)
        u1 = X1.mean(0, keepdims=True)  # 计算正类样本的均值

        # 计算类内散度矩阵 Sw
        sw = np.dot((X0 - u0).T, X0 - u0) + np.dot((X1 - u1).T, X1 - u1)
        # 计算线性判别分析的权重 w
        #np.linalg.inv是用来计算逆矩阵，reshape(1,-1)是重塑行向量
        w = np.dot(np.linalg.inv(sw), (u0 - u1).T).reshape(1, -1)  # (1, n)

        if plot_:  # 如果需要绘图
            fig, ax = plt.subplots()  # 创建一个子图

            #解决无法正常显示标签
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
            plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
            ax.spines['right'].set_color('none')  # 移除右边框
            ax.spines['top'].set_color('none')  # 移除顶部边框
            ax.spines['left'].set_position(('data', 0))  # 将左边框移到y轴0位置
            ax.spines['bottom'].set_position(('data', 0))  # 将底边框移到x轴0位置

            plt.scatter(X1[:, 0], X1[:, 1], c='k', marker='o', label='good')  # 绘制正类样本
            plt.scatter(X0[:, 0], X0[:, 1], c='r', marker='x', label='bad')  # 绘制负类样本

            plt.xlabel('密度', labelpad=1)  # 设置x轴标签
            plt.ylabel('含糖量')  # 设置y轴标签
            plt.legend(loc='upper right')  # 显示图例

            #绘制直线y=wx
            x_tmp = np.linspace(-0.05, 0.15)  # 创建用于绘制直线的x值
            #绘制向量w对应的直线
            y_tmp = x_tmp * w[0, 1] / w[0, 0]  # 根据权重计算对应的y值
            plt.plot(x_tmp, y_tmp, '#808080', linewidth=1)  # 绘制决策边界

            #归一化后方便计算接下来的投影矩阵
            wu = w / np.linalg.norm(w)  # 规范化权重向量

            # 投影负类样本
            # 利用np.dot(wu.T, wu)计算投影矩阵
            #本来根据投影公式应该是投影矩阵左乘，但是由于采取的是含向量的形式，因而右乘
            X0_project = np.dot(X0, np.dot(wu.T, wu))  # 将负类样本投影到向量w上
            #根据投影点的x和y坐标进行绘制
            plt.scatter(X0_project[:, 0], X0_project[:, 1], c='r', s=15)  # 绘制负类样本的投影
            #为原始点和投影点绘制虚线
            for i in range(X0.shape[0]):
                plt.plot([X0[i, 0], X0_project[i, 0]], [X0[i, 1], X0_project[i, 1]], '--r', linewidth=1)

            # 投影正类样本
            X1_project = np.dot(X1, np.dot(wu.T, wu))  # 计算正类样本的投影
            plt.scatter(X1_project[:, 0], X1_project[:, 1], c='k', s=15)  # 绘制正类样本的投影
            for i in range(X1.shape[0]):  # 为每个正类样本绘制虚线到投影点
                plt.plot([X1[i, 0], X1_project[i, 0]], [X1[i, 1], X1_project[i, 1]], '--k', linewidth=1)

            # 绘制均值点的投影
            u0_project = np.dot(u0, np.dot(wu.T, wu))  # 计算负类均值的投影
            plt.scatter(u0_project[:, 0], u0_project[:, 1], c='#FF4500', s=60)  # 绘制负类均值的投影
            u1_project = np.dot(u1, np.dot(wu.T, wu))  # 计算正类均值的投影
            plt.scatter(u1_project[:, 0], u1_project[:, 1], c='#696969', s=60)  # 绘制正类均值的投影

            # 注释负类均值的投影
            ax.annotate(r'u0 投影点',
                        xy=(u0_project[:, 0], u0_project[:, 1]),
                        xytext=(u0_project[:, 0] - 0.2, u0_project[:, 1] - 0.1),
                        size=13,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="->",
                                        color="k",
                                        )
                        )

            # 注释正类均值的投影
            ax.annotate(r'u1 投影点',
                        xy=(u1_project[:, 0], u1_project[:, 1]),
                        xytext=(u1_project[:, 0] - 0.1, u1_project[:, 1] + 0.1),
                        size=13,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="->",
                                        color="k",
                                        )
                        )
            plt.axis("equal")  # 保持x轴和y轴的单位刻度长度一致
            plt.show()  # 显示绘图

        self.w = w  # 保存权重向量
        self.u0 = u0  # 保存负类均值
        self.u1 = u1  # 保存正类均值
        return self  # 返回当前对象

    def predict(self, X):  # 定义预测函数，接受特征X
        project = np.dot(X, self.w.T)  # 计算投影

        wu0 = np.dot(self.w, self.u0.T)  # 计算负类均值在权重方向上的投影
        wu1 = np.dot(self.w, self.u1.T)  # 计算正类均值在权重方向上的投影

        # 预测样本属于正类或负类
        return (np.abs(project - wu1) < np.abs(project - wu0)).astype(int)  # 返回预测结果（1或0）


if __name__ == '__main__':  # 如果是主程序
    data_path = r'C:\Users\叶枫\Desktop\MachineLearning_Zhouzhihua_ProblemSets\ch3--线性模型\3.3\watermelon3_0_Ch.csv'  # 数据文件路径

    data = pd.read_csv(data_path).values  # 读取CSV文件并转换为NumPy数组

    X = data[:, 7:9].astype(float)  # 提取特征（第7列到第8列）
    y = data[:, 9]  # 提取标签（第9列）

    y[y == '是'] = 1  # 将标签'是'转换为1
    y[y == '否'] = 0  # 将标签'否'转换为0
    y = y.astype(int)  # 将标签转换为整型

    lda = LDA()  # 实例化LDA对象
    lda.fit(X, y, plot_=True)  # 拟合模型并绘制结果
    print(lda.predict(X))  # 输出预测结果，应该与逻辑回归的结果一致
    print(y)  # 输出真实标签
