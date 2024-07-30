import numpy as np

class LayerNormalization:
    def __init__(self, epsilon=1e-5):
        # 初始化LayerNormalization类，设定epsilon的默认值
        self.epsilon = epsilon  # 防止分母为零的小常数
        self.gamma = None  # 缩放参数
        self.beta = None  # 平移参数

    def initialize_params(self, D):
        # 初始化gamma和beta的参数
        self.gamma = np.ones(D)  # 初始化缩放参数为1
        self.beta = np.zeros(D)  # 初始化平移参数为0

    def forward(self, X):
        # 前向传播，X是输入数据
        if self.gamma is None or self.beta is None:
            # 如果是第一次运行，初始化参数
            self.initialize_params(X.shape[1])  # X.shape[1]是特征的维度

        # 计算每个样本的均值和方差
        mean = np.mean(X, axis=1, keepdims=True)  # keepdims=True保持结果的维度一致
        var = np.var(X, axis=1, keepdims=True)  # keepdims=True保持结果的维度一致

        # 标准化输入数据
        X_normalized = (X - mean) / np.sqrt(var + self.epsilon)

        # 应用可学习的缩放和平移参数
        out = self.gamma * X_normalized + self.beta
        return out

    def __call__(self, X):
        # 使类实例可以像函数一样被调用
        return self.forward(X)

# 示例数据
np.random.seed(0)  # 设置随机种子以确保结果可重复
X = np.random.randn(10, 5)  # 生成一个随机的10x5的矩阵

# 创建LayerNormalization实例
ln = LayerNormalization()

# 进行前向传播
output = ln(X)
print("Layer Normalization Output:\n", output)  # 打印输出
