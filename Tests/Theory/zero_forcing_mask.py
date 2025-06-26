import numpy as np
import torch
from torch.func import vmap


def safe_masked_zf0(H, y, mask, No, eps=1e-6):
    """
    Zero-Forcing 解，忽略 mask == 0 的列
    H: [B, M, N]
    y: [B, M]
    mask: [B, N] (0 表示该位置 x 为 0，不需要估计)
    返回: x_hat: [B, N]
    """

    def masked_zf_single(H, y, mask):
        # 用 mask 扩展到 H 的形状：[M, N]
        
        # 避免不可逆：加一点正则项
        #HTH = H_masked.T @ H_masked  # [N, N]
        #HTH = H.T @ H @ torch.diag(mask) # [N, N]
        HTH = (H.T.conj() @ H) * mask[None, ...] # mask [(B), 1, N]
        
        HTH += No * torch.eye(HTH.shape[0], device=H.device)
        
        #HTH0 = torch.diag(mask) @ HTH

        H_pinv = torch.linalg.solve(HTH, H.T.conj())  # [N, M]
        x_hat = (mask[..., None] * H_pinv) @ y  # [N]

        # 由于被 mask 的位置本来就是 0，保留 x_hat 即可
        return x_hat

    return vmap(masked_zf_single, in_dims=(0, 0, 0))(H, y, mask)

def safe_masked_zf(H, y, mask, No, eps=1e-6):
    """
    Zero-Forcing 解，忽略 mask == 0 的列
    H: [B, M, N]
    y: [B, M]
    mask: [B, N] (0 表示该位置 x 为 0，不需要估计)
    返回: x_hat: [B, N]
    """

    def masked_zf_single(H, y, mask):
        # 用 mask 扩展到 H 的形状：[M, N]
        H_masked = H * mask.unsqueeze(0)  # [M, N]
        mask = mask.to(torch.complex64)
        
        # 避免不可逆：加一点正则项
        #HTH = H_masked.T @ H_masked  # [N, N]
        HTH = H.T.conj() @ H @ torch.diag(mask) # [N, N]
        
        
        HTH += No * torch.eye(HTH.shape[0], device=H.device)

        H_pinv = torch.linalg.solve(HTH, H_masked.T.conj())  # [N, M]
        x_hat = torch.diag(mask) @H_pinv @ y  # [N]

        # 由于被 mask 的位置本来就是 0，保留 x_hat 即可
        return x_hat

    return vmap(masked_zf_single, in_dims=(0, 0, 0))(H, y, mask)

def safe_masked_zf2(H, y, mask, eps=1e-6):
    """
    Zero-Forcing 解，忽略 mask == 0 的列
    H: [B, M, N]
    y: [B, M]
    mask: [B, N] (0 表示该位置 x 为 0，不需要估计)
    返回: x_hat: [B, N]
    """

    def masked_zf_single(H, y, mask):
        # 用 mask 扩展到 H 的形状：[M, N]
        H_masked = H * mask.unsqueeze(0)  # [M, N]
        
        # 避免不可逆：加一点正则项
        HTH = H_masked.T.conj() @ H_masked  # [N, N]
        
        eigenvalues = torch.linalg.eigvalsh(HTH)
        lambda_max = eigenvalues[-1]
        epsilon = eps * lambda_max
        
        HTH += eps * torch.eye(HTH.shape[0], device=H.device)

        H_pinv = torch.linalg.solve(HTH, H_masked.T.conj())  # [N, M]
        x_hat = H_pinv @ y  # [N]

        # 由于被 mask 的位置本来就是 0，保留 x_hat 即可
        return x_hat

    return vmap(masked_zf_single, in_dims=(0, 0, 0))(H, y, mask)

def safe_masked_zf3(H, y, mask, eps=1e-6):
    """
    Zero-Forcing 解，忽略 mask == 0 的列
    H: [B, M, N]
    y: [B, M]
    mask: [B, N] (0 表示该位置 x 为 0，不需要估计)
    返回: x_hat: [B, N]
    """

    def masked_zf_single(H, y, mask):
        # 用 mask 扩展到 H 的形状：[M, N]
        H_masked = H * mask.unsqueeze(0)  # [M, N]
        
        # 避免不可逆：加一点正则项
        HTH = H_masked.T @ H_masked  # [N, N]
        
        eigenvalues = torch.linalg.eigvalsh(HTH)
        lambda_max = eigenvalues[-1]
        epsilon = eps * lambda_max
        
        HTH += epsilon * torch.eye(HTH.shape[0], device=H.device)

        H_pinv = torch.linalg.solve(HTH, H_masked.T)  # [N, M]
        x_hat = H_pinv @ y  # [N]

        # 由于被 mask 的位置本来就是 0，保留 x_hat 即可
        return x_hat

    return vmap(masked_zf_single, in_dims=(0, 0, 0))(H, y, mask)


# 随机构造输入
mean0 = 0.0;
mean1 = 0.0;
mean2 = 1.0;
mean3 = 1.0;
No = 0.0001;
for i in range(int(1e4)):
    B, M, N = 4, 10, 8
    H = torch.randn(B, M, N) + torch.randn(B, M, N)*1j
    x_true = torch.randn(B, N) + torch.randn(B, N)*1j
    mask = (torch.rand(B, N) > 0.4).float()  # 每个样本有部分 0
    x_true = x_true * mask  # 强制这些维度为 0
    n = np.sqrt(No/2)*torch.randn(B, M)
    y = (H @ x_true.unsqueeze(-1)).squeeze(-1) + n
    
    # 调用 ZF 解法
    x_hat0 = safe_masked_zf0(H, y, mask, No)
    x_hat = safe_masked_zf(H, y, mask, No)
    x_hat2 = safe_masked_zf2(H, y, mask)
    #x_hat3 = safe_masked_zf3(H, y, mask)
    
    err0 = torch.sum(x_hat0*(1-mask));
    err1 = torch.sum(x_hat*(1-mask));
    err2 = torch.sum(x_hat2*(1-mask));
    
    if err1 != 0+0j:
        raise Exception("error detect");
    if err2 != 0+0j:
        raise Exception("error detect");
    
    mean0 += torch.mean((x_hat0 - x_true) ** 2).item()
    mean1 += torch.mean((x_hat - x_true) ** 2).item()
    mean2 += torch.mean((x_hat2 - x_true) ** 2).item()
    #mean3 += torch.mean((x_hat3 - x_true) ** 2).item()

# 检查结果
print("MSE0:", mean0/1e4)
print("MSE1:", mean1/1e4)
print("MSE2:", mean2/1e4)
#print("MSE3:", mean3/1e4)

x_hat = x_hat.numpy();
x_true = x_true.numpy();
x_hat_diff = abs(x_hat - x_true)