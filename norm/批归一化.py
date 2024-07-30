import numpy as np

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum=0.9):
        # 初始化BatchNormalization类，设定epsilon和momentum的默认值
        self.epsilon = epsilon  # 防止分母为零的小常数
        self.momentum = momentum  # 用于更新运行中均值和方差的动量
        self.running_mean = None  # 运行中的均值
        self.running_var = None  # 运行中的方差
        self.gamma = None  # 缩放参数
        self.beta = None  # 平移参数

    def initialize_params(self, D):
        # 初始化gamma, beta, running_mean和running_var的参数
        self.gamma = np.ones(D)  # 初始化缩放参数为1
        self.beta = np.zeros(D)  # 初始化平移参数为0
        self.running_mean = np.zeros(D)  # 初始化运行中的均值为0
        self.running_var = np.ones(D)  # 初始化运行中的方差为1

    def forward(self, X, training=True):
        # 前向传播，X是输入数据，training表示是否为训练模式
        if self.running_mean is None:
            # 如果是第一次运行，初始化参数
            self.initialize_params(X.shape[1])  # X.shape[1]是特征的维度
        
        if training:
            # 训练模式下
            batch_mean = np.mean(X, axis=0)  # 计算mini-batch的均值
            batch_var = np.var(X, axis=0)  # 计算mini-batch的方差
            # axis=0 代表沿着第一维，也就是行，可以想象成一个尺子，和每行对齐，依次往下移动，得到所有行在这些列上的均值，因此维度是【5】
            # axis=1 代表沿着第二维，也就是列。同理，最后维度是行的维度。
            # 更新运行中的均值
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            # 更新运行中的方差
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # 标准化输入数据
            X_normalized = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            # 测试模式下，使用运行中的均值和方差进行标准化
            X_normalized = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 应用可学习的缩放和平移参数
        out = self.gamma * X_normalized + self.beta
        return out

    def __call__(self, X, training=True):
        # 使类实例可以像函数一样被调用
        return self.forward(X, training)

# 示例数据
np.random.seed(0)  # 设置随机种子以确保结果可重复
X = np.random.randn(10, 5)  # 生成一个随机的10x5的矩阵

# 创建BatchNormalization实例
bn = BatchNormalization()

# 在训练模式下进行前向传播
output_train = bn(X, training=True)
print("Training Mode Output:\n", output_train)  # 打印训练模式下的输出

# 在测试模式下进行前向传播
output_test = bn(X, training=False)
print("Testing Mode Output:\n", output_test)  # 打印测试模式下的输出
