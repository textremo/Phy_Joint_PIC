import torch

# 假设 H_dense 是稀疏矩阵
H_dense = torch.tensor([
    [0, 0, 1],
    [2, 0, 0],
    [0, 3, 0],
], dtype=torch.float32)

# 转换成 COO 格式稀疏矩阵
indices = torch.nonzero(H_dense).t()   # [2, nnz]
values = H_dense[H_dense != 0]
H_sparse = torch.sparse_coo_tensor(indices, values, H_dense.size())

# 稀疏矩阵乘法
x = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)  # shape (3,1)
y = torch.sparse.mm(H_sparse, x)
print(y)