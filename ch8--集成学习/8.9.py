import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import cohen_kappa_score

# 设置中文字体
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 这里使用的是黑体字体，你可以根据需要更换

# 数据集
X = np.array([[0.697, 0.46], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
              [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
              [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.36, 0.37],
              [0.593, 0.042], [0.719, 0.103]])
Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

# 训练 AdaBoost 模型
ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=20, random_state=42)
ada_model.fit(X, Y)

# 训练 Bagging 模型
bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=20, random_state=42)
bag_model.fit(X, Y)

# 计算多样性度量的函数
def diversity_metric(models, X):
    n_models = len(models)  # 模型数量
    n_samples = X.shape[0]  # 样本数量
    predictions = np.zeros((n_models, n_samples))  # 存储每个模型的预测结果

    # 获取每个模型的预测结果
    for i, model in enumerate(models):
        predictions[i] = model.predict(X)

    diversity = 0
    # 计算多样性度量
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diversity += np.mean(predictions[i] != predictions[j])

    diversity /= (n_models * (n_models - 1) / 2)
    return diversity

# 提取 AdaBoost 和 Bagging 的个体模型
ada_models = ada_model.estimators_
bag_models = bag_model.estimators_

# 计算多样性
ada_diversity = diversity_metric(ada_models, X)
bag_diversity = diversity_metric(bag_models, X)

# 计算 κ-误差的函数
def kappa_error(models, X, Y):
    n_models = len(models)  # 模型数量
    kappas = []  # 存储 kappa 值
    errors = []  # 存储错误率

    # 计算每个模型的 kappa 值和错误率
    for model in models:
        predictions = model.predict(X)
        kappas.append(cohen_kappa_score(Y, predictions))
        errors.append(np.mean(predictions != Y))

    return kappas, errors

# 计算 AdaBoost 和 Bagging 的 κ-误差
ada_kappas, ada_errors = kappa_error(ada_models, X, Y)
bag_kappas, bag_errors = kappa_error(bag_models, X, Y)

# 绘制多样性度量图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(['AdaBoost', 'Bagging'], [ada_diversity, bag_diversity])
plt.title('多样性度量', fontproperties=font)
plt.ylabel('多样性', fontproperties=font)

# 绘制 κ-误差图
plt.subplot(1, 2, 2)
plt.scatter(ada_kappas, ada_errors, label='AdaBoost', color='blue')
plt.scatter(bag_kappas, bag_errors, label='Bagging', color='green')
plt.title('κ-误差图', fontproperties=font)
plt.xlabel('Cohen\'s Kappa', fontproperties=font)
plt.ylabel('错误率', fontproperties=font)
plt.legend(prop=font)

plt.tight_layout()
plt.show()