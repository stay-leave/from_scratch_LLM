import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self, input_dim) -> None:
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim

        self.W_Q = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)  # 通常指定一个非零的 p

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim] -> [2, 3, 4]
        batch_size = x.size(0)

        q = self.W_Q(x)  # [2, 3, 4]
        k = self.W_K(x)  # [2, 3, 4]
        v = self.W_V(x)  # [2, 3, 4]

        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.input_dim)  # [2, 3, 3]
        weight = self.softmax(scores)  # [2, 3, 3]
        weight = self.dropout(weight)  # 在注意力权重上应用 Dropout

        # 计算加权和
        output = torch.matmul(weight, v)  # [2, 3, 4]

        return output, weight  # 返回输出和注意力权重

# 示例用法
# 一个示例三维张量，形状为 (2, 3, 4)
tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0]],

                       [[13.0, 14.0, 15.0, 16.0],
                        [17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0]]])

# 初始化自注意力模块
attention = SelfAttention(tensor.size(-1))

# 前向传播
output, weights = attention(tensor)

print(output)
print(weights)
