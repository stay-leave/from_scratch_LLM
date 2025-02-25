import torch
import torch.nn as nn

# 专家
class ExpertNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size # 隐藏层维度
        self.intermediate_size = intermediate_size # 中间层维度

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        output = self.linear2(x)
        return output

# 路由
class Router(nn.Module):
    def __init__(self, hidden_size, expert_num, top_k):
        super().__init__()
        self.router = nn.Linear(hidden_size, expert_num)
        self.top_k = top_k
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        x = self.router(x)
        x = nn.functional.softmax(x, dim=-1) # 门控网络
        topk_weight, topk_idx = torch.topk(x, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        return topk_weight, topk_idx

class MOELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, expert_num, top_k):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_num = expert_num
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [ExpertNetwork(self.hidden_size, self.intermediate_size) for _ in range(self.expert_num)]
        )
        self.router = Router(self.hidden_size, self.expert_num, self.top_k)

    def forward(self, x):
        # shape of x is (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = x.size()
        token_num = batch_size * seq_len
        x_flat = x.view(token_num, self.hidden_size)
        topk_weight, topk_idx = self.router(x)
        output = torch.zeros_like(x_flat)
        for token_idx in range(token_num):
            for expert_idx in range(self.top_k):
                expert = self.experts[topk_idx[token_idx, expert_idx]]
                output[token_idx] += topk_weight[token_idx, expert_idx] * expert(x_flat[token_idx])
        output = output.view(batch_size, seq_len, self.hidden_size)
        return output

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
EXPERT_NUM = 8
TOP_K = 2

inputs = torch.randn(2, 11, 4096)
moe_layer = MOELayer(HIDDEN_SIZE, INTERMEDIATE_SIZE, EXPERT_NUM, TOP_K)
outputs = moe_layer(inputs)
print(outputs.size())