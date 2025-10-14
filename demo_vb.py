import numpy as np

def vb_jed_rayleigh(Y, X_pilot, M, K, max_iter=50, tol=1e-6):
    """
    Variational Bayes for Joint Channel Estimation and Detection (MIMO, Rayleigh fading)
    
    Args:
        Y: 接收信号 (M×1) 向量
        X_pilot: 已知导频 (K×1)
        M: 接收天线数
        K: 发射天线数
        max_iter: 最大迭代次数
        tol: 收敛阈值
    Returns:
        H_mean: 估计的 MIMO 信道矩阵 (M×K)
        x_est: 估计的符号向量 (K×1)
    """

    # -----------------------------
    # 初始化
    # -----------------------------
    np.random.seed(0)
    # 噪声精度 (gamma)
    gamma = 1.0
    # 信道先验
    sigma_h2 = 1.0
    # 符号先验
    sigma_x2 = 1.0

    # 初始化 H 与 x
    H_mean = np.random.randn(M, K) + 1j * np.random.randn(M, K)
    H_mean /= np.sqrt(2*K)
    x_mean = np.random.randn(K, 1) + 1j * np.random.randn(K, 1)
    x_mean /= np.sqrt(2)

    H_cov = np.eye(K, dtype=complex)
    x_cov = np.eye(K, dtype=complex)

    # -----------------------------
    # VB 迭代
    # -----------------------------
    for it in range(max_iter):

        # (1) 更新 q(H)
        H_cov_inv = gamma * (x_mean @ x_mean.conj().T + x_cov) + (1 / sigma_h2) * np.eye(K)
        H_cov = np.linalg.inv(H_cov_inv)
        H_mean = gamma * (Y @ x_mean.conj().T) @ H_cov

        # (2) 更新 q(x)
        X_cov_inv = gamma * (H_mean.conj().T @ H_mean + np.trace(H_cov) * np.eye(K)) + (1 / sigma_x2) * np.eye(K)
        x_cov = np.linalg.inv(X_cov_inv)
        x_new = gamma * x_cov @ H_mean.conj().T @ Y

        # (3) 更新 gamma (噪声精度)
        Y_hat = H_mean @ x_new
        resid = np.linalg.norm(Y - Y_hat)**2
        gamma_new = (M + K) / (resid + np.trace(H_mean.conj().T @ H_mean @ x_cov))

        # 阻尼 + 收敛
        alpha = 0.7
        x_mean = alpha * x_new + (1 - alpha) * x_mean
        gamma = alpha * gamma_new + (1 - alpha) * gamma

        if np.linalg.norm(x_new - x_mean) < tol:
            break

    return H_mean, x_mean


# -----------------------------
# 示例：Rayleigh MIMO 系统
# -----------------------------
if __name__ == "__main__":
    M, K = 4, 2  # 4 接收天线, 2 发射天线

    # 生成 Rayleigh 信道
    H_true = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)

    # 发送符号 (QPSK)
    X_true = (np.random.choice([1, -1], (K, 1)) + 1j * np.random.choice([1, -1], (K, 1))) / np.sqrt(2)

    # 噪声
    snr_db = 20
    snr = 10**(snr_db / 10)
    noise_var = 1 / snr
    noise = np.sqrt(noise_var/2) * (np.random.randn(M, 1) + 1j * np.random.randn(M, 1))

    # 接收信号
    Y = H_true @ X_true + noise

    # VB 联合估计与检测
    H_est, X_est = vb_jed_rayleigh(Y, X_pilot=X_true, M=M, K=K)

    print("True x:\n", X_true)
    print("Estimated x:\n", np.round(X_est, 3))
    print("\nTrue H:\n", np.round(H_true, 3))
    print("Estimated H:\n", np.round(H_est, 3))