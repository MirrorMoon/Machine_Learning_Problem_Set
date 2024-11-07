import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

# 生成一个复杂且带有噪声的数据集
# n_samples: 样本数量
# n_features: 特征数量
# n_informative: 有效特征数量
# n_redundant: 冗余特征数量
# n_clusters_per_class: 每个类别的簇数量
# flip_y: 标签噪声比例
# random_state: 随机种子，保证结果可重复
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, n_clusters_per_class=1, flip_y=0.4, random_state=42)

# 将数据集划分为训练集和测试集
# test_size: 测试集所占比例
# random_state: 随机种子，保证结果可重复
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义 k-NN（K-最近邻）分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 训练 k-NN 模型
knn.fit(X_train, y_train)

# 使用训练好的 k-NN 模型对测试集进行预测
y_pred_knn = knn.predict(X_test)

# 计算 k-NN 模型的准确率
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# 使用 Bagging 方法提升 k-NN 分类器的性能
# estimator: 基础分类器
# n_estimators: 基础分类器的数量
# random_state: 随机种子，保证结果可重复
bagging_knn = BaggingClassifier(estimator=knn, n_estimators=50, random_state=42)

# 训练 Bagging k-NN 模型
bagging_knn.fit(X_train, y_train)

# 使用训练好的 Bagging k-NN 模型对测试集进行预测
y_pred_bagging_knn = bagging_knn.predict(X_test)

# 计算 Bagging k-NN 模型的准确率
accuracy_bagging_knn = accuracy_score(y_test, y_pred_bagging_knn)

# 绘制 k-NN 和 Bagging k-NN 模型准确率的比较图表
labels = ['k-NN', 'Bagging k-NN']
accuracies = [accuracy_knn, accuracy_bagging_knn]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, accuracies, color=['blue', 'green'])
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('k-NN 和 Bagging k-NN 的比较')
plt.ylim(0, 1)

# 在柱状图上添加准确率值
for bar, accuracy in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{accuracy:.2f}', ha='center', va='bottom')

plt.show()

# 输出 k-NN 和 Bagging k-NN 模型的准确率
print(f"k-NN 准确率: {accuracy_knn:.4f}")
print(f"Bagging k-NN 准确率: {accuracy_bagging_knn:.4f}")