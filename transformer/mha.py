import torch 
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()

        self.d_model = d_model  # 输入的维度，例如 16
        self.num_heads = num_heads  # 注意力头数，例如 8
        # 检查头数和总维度数是否能整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 每个头的维度数
        self.d_head = d_model // num_heads  # 每个头的维度，例如 2
        # 初始化三个输入的权重矩阵
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        # 初始化 softmax
        self.softmax = nn.Softmax(-1)  # 在最后一个维度上执行 softmax
        self.dropout = nn.Dropout(0.1)  # Dropout 概率为 0.1
        # 全连接层
        self.fc = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, mask=None):
        """计算每个头的结果"""
        # 点积计算相似度分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # mask，屏蔽某些位置的分数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax 归一化
        scores = self.softmax(scores)
        # 应用 Dropout
        scores = self.dropout(scores)
        # 乘以值向量
        output = torch.matmul(scores, v)

        return output, scores

    def forward(self, q, k, v, mask=None):
        # 得到批次大小
        batch_size = q.size(0)  # 例如 2
        # 权重矩阵转换，形状转换到合适的维度
        q = self.w_qs(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # [2,5,16] -> [2,8,5,2]
        k = self.w_ks(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # 同上
        v = self.w_vs(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)  # 同上
        # 计算所有头的注意力输出
        attention_output, weight = self.attention(q, k, v, mask)  # [2,8,5,2]
        # 重新组合多头结果
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [2,5,16]
        # 通过全连接层得到最终输出
        attention_output = self.fc(attention_output)  # [2,5,16]

        return attention_output, weight

if __name__ == "__main__":
    x = torch.randn([2, 5, 16])

    # 初始化多头注意力机制
    attention = MHA(x.size(-1), 8)
    print(attention)
        
    # 前向传播
    q = torch.randn([2, 5, 16])
    k = torch.randn([2, 5, 16])
    v = torch.randn([2, 5, 16])
    out, weight = attention(q, k, v)

    print(out.shape)  # 应为 [2, 5, 16]
    
    # 反向传播
    loss = out.mean()
    loss.backward()
    print("backward over!")
