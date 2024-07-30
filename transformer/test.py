import torch
import math

# 设置随机种子，确保结果可复现
torch.manual_seed(42)

# 初始化输入数据
batch_size, seq_len, embedding_size = 16, 20, 30
x = torch.randn([batch_size, seq_len, embedding_size])




