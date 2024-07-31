import torch
import torch.nn as nn
import math

class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super(FeedForward, self).__init__()
        self.up = nn.Linear(input_dim, 4 * input_dim)  # 上采样线性层
        self.down = nn.Linear(4 * input_dim, input_dim)  # 下采样线性层
        self.relu = nn.ReLU()  # ReLU 激活函数
    
    def forward(self, x):
        up_output = self.up(x)  # 上采样
        activated = self.relu(up_output)  # 激活函数
        out = self.down(activated)  # 下采样
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # 初始化可学习参数 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / (std + self.eps)
        y = self.gamma * x_hat + self.beta
        return y

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MHA, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert self.d_head * num_heads == self.d_model, "d_model must be divisible by num_heads"

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        q = self.W_Q(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weight = self.softmax(scores)
        weight = self.dropout(weight)
        attention_output = torch.matmul(weight, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention_output)
        return output, weight

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        pe = torch.zeros(config.seq_len, config.d_model)
        position = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attention = MHA(d_model, num_heads)
        self.layer_norm1 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头自注意力
        attention_output, _ = self.self_attention(x, mask)
        # 残差连接和层归一化
        x = self.layer_norm1(x + self.dropout(attention_output))
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        # 残差连接和层归一化
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

# 示例用法
class Config:
    seq_len = 10
    d_model = 512
    dropout = 0.1

config = Config()
pos_encoding = PositionalEncoding(config)
encoder = TransformerEncoder(d_model=config.d_model, num_heads=8, ff_hidden_dim=2048, dropout=0.1)

# 示例输入张量，形状为 (batch_size, seq_len, d_model)
x = torch.randn(2, config.seq_len, config.d_model)

# 加入位置编码
x = pos_encoding(x)

# 前向传播
out = encoder(x)

print("输出形状:", out.shape)  # 输出: (2, 10, 512)
