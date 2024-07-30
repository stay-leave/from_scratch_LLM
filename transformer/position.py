import torch
import torch.nn as nn
import math

'''位置编码模块'''
class PositionalEncoding(nn.Module):
    # 在输入上加入了位置编码

    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        self.dropout = nn.Dropout(p=config.dropout)
        # 初始化位置编码矩阵
        pe = torch.zeros(config.seq_len, config.d_model) # [20,30]
        position = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1) # [20,1] # 位置索引
        # 计算频率矩阵
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model)) # [15]
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置使用cos
        
        # 添加一个维度，以匹配批次大小的维度
        pe = pe.unsqueeze(0) # [1, 20, 30]
        # 将位置编码矩阵 pe 注册为模型的一个持久缓冲区。
        # 这意味着 pe 不会被视为模型的可训练参数（不会在训练过程中更新），但会作为模型的一部分进行保存和加载。
        # 这样做的好处是可以将固定的值（如位置编码）与模型一起保存，而不会被优化器更新。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 从位置编码矩阵 pe 中提取与输入 x 的序列长度相匹配的部分。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        return self.dropout(x)