import torch
import torch.nn as nn

# 假设有一个批次(batch)的嵌入向量
batch, sentence_length, embedding_dim = 20, 5, 10

# 生成随机的嵌入向量，形状为 [20, 5, 10]
embedding = torch.randn(batch, sentence_length, embedding_dim)

# 创建RMSNorm实例，传入的特征维度是 embedding_dim
rms_norm = nn.RMSNorm(embedding_dim)

# 对嵌入向量进行归一化
normalized_embedding = rms_norm(embedding)

print("Original Embedding:\n", embedding.shape)
print("Normalized Embedding:\n", normalized_embedding.shape)
