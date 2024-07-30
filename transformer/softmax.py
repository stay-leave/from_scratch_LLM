import torch

def softmax(tensor, dim=-1):
    """
    对三维张量的指定维度应用 Softmax。
    
    参数:
    - tensor: 输入张量，形状为 (batch_size, seq_len, num_features) 的三维张量。
    - dim: 进行 Softmax 的维度，默认值为 -1（最后一个维度）。
    
    返回:
    - 经过 Softmax 变换后的张量。
    """
    # 对指定维度上的最大值进行减法操作，避免数值溢出
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    stable_tensor = tensor - max_vals
    
    # 计算 exp 值
    exp_tensor = torch.exp(stable_tensor)
    
    # 计算 softmax
    sum_exp = exp_tensor.sum(dim=dim, keepdim=True)
    softmax_tensor = exp_tensor / sum_exp
    
    return softmax_tensor

# 示例用法
if __name__ == "__main__":
    # 一个示例三维张量，形状为 (2, 3, 4)
    tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]],

                           [[13.0, 14.0, 15.0, 16.0],
                            [17.0, 18.0, 19.0, 20.0],
                            [21.0, 22.0, 23.0, 24.0]]])

    # 在最后一个维度上应用 Softmax
    softmax_output = softmax(tensor, dim=-1)
    print(softmax_output)
