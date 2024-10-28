import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


# 定义一个简单的全连接神经网络
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 定义四个全连接层
        self.fc1 = torch.nn.Linear(28 * 28, 64)  # 输入层，将28x28的图片展平成784个输入节点，输出为64维
        self.fc2 = torch.nn.Linear(64, 64)  # 第二层，输入64维，输出64维
        self.fc3 = torch.nn.Linear(64, 64)  # 第三层，输入64维，输出64维
        self.fc4 = torch.nn.Linear(64, 10)  # 输出层，10个输出节点对应10个类别

    # 定义前向传播过程
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))  # 第一个全连接层+ReLU激活
        x = torch.nn.functional.relu(self.fc2(x))  # 第二个全连接层+ReLU激活
        x = torch.nn.functional.relu(self.fc3(x))  # 第三个全连接层+ReLU激活
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)  # 最后一层+log_softmax用于分类
        return x


# 数据加载器函数，返回训练或测试数据集的DataLoader对象
def get_data_loader(is_train):
    # 数据预处理，转换为Tensor
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载MNIST数据集，下载数据并转换格式
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 使用DataLoader包装数据集，每次批量加载15张图像
    return DataLoader(data_set, batch_size=15, shuffle=True)


# 评估函数，计算模型在测试集上的准确率
def evaluate(test_data, net):
    n_correct = 0  # 记录正确预测的数量
    n_total = 0  # 记录总的样本数量
    with torch.no_grad():  # 在评估时不需要计算梯度
        for (x, y) in test_data:
            # 前向传播，x.view将输入展平为28x28=784维的向量
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:  # 比较预测标签和真实标签
                    n_correct += 1
                n_total += 1
    return n_correct / n_total  # 返回准确率


# 主函数
def main():
    # 加载训练数据和测试数据
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    # 实例化网络
    net = Net()

    # 初始模型准确率
    print("initial accuracy:", evaluate(test_data, net))
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练2个epoch
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad()  # 将模型的梯度置为零
            output = net.forward(x.view(-1, 28 * 28))  # 前向传播
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失（负对数似然损失）
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 优化器更新权重
        # 每个epoch后评估模型准确率
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 可视化测试集中的前4张图片及其预测
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))  # 对每张图片进行预测
        plt.figure(n)  # 创建新的图形窗口
        plt.imshow(x[0].view(28, 28))  # 显示图片
        plt.title("prediction: " + str(int(predict)))  # 显示预测结果
    plt.show()  # 显示所有图片


# 如果文件作为主程序运行，则调用main()函数
if __name__ == "__main__":
    main()
