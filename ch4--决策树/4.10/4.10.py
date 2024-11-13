import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 读取数据
# file_path: 数据文件的路径
file_path = r'D:\Machine_Learning_Problem_Set\ch4--决策树\watermelon3_0_Ch.csv'
data = pd.read_csv(file_path)

# 提取特征和标签
# X: 特征数据
# y: 标签数据
X = data.loc[:, '色泽':'含糖率']
y = data['好瓜']

# 编码类别特征
# 将类别特征转换为数值型
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# 分割数据集
# 将数据集分为训练集和测试集
# test_size: 测试集所占比例
# random_state: 随机种子，保证结果可重复
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 确定LDA的n_components参数
# n_classes: 类别数量
# n_features: 特征数量
# n_components: LDA降维后的特征数量，不能超过类别数减一和特征数的最小值
n_classes = len(y.unique())
n_features = X.shape[1]
n_components = min(n_features, n_classes - 1)

# 创建一个包含LDA变换和决策树分类的流水线
# model: 包含LDA和决策树的流水线模型
model = Pipeline([
    ('lda', LDA(n_components=n_components)),  # 将特征降至合适的组合特征数量
    ('tree', DecisionTreeClassifier(random_state=42))  # 决策树分类器
])

# 训练模型
# 使用训练集数据训练流水线模型
model.fit(X_train, y_train)

# 测试模型
# 使用测试集数据进行预测
y_pred = model.predict(X_test)

# 计算模型的准确率
# accuracy: 模型的预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")