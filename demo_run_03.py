import numpy as np

qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

# ========== 参数设置 ==========
M = 4          # 接收天线数
K = 2          # 发射天线数
T = 1         # 时隙长度
pilot_len = 4  # 导频长度
snr_db = 15

# ========== 生成导频 + 数据 ==========
np.random.seed(0)
pilot = (np.random.randn(K, pilot_len) + 1j*np.random.randn(K, pilot_len)) / np.sqrt(2)
data  = np.sign(np.random.randn(K, T - pilot_len))  # BPSK 数据
x_true = np.concatenate([pilot, data], axis=1)      # 拼成完整发射矩阵

# ========== 生成 Rayleigh MIMO 信道 ==========
H_true = (np.random.randn(M, K) + 1j*np.random.randn(M, K)) / np.sqrt(2)

# ========== 接收信号 ==========
snr = 10 ** (snr_db / 10)
noise_var = 1 / snr
N = np.sqrt(noise_var/2) * (np.random.randn(M, T) + 1j*np.random.randn(M, T))
Y = H_true @ x_true + N

# ========== Step 1: 初始信道估计（基于导频）==========
X_pilot = pilot
Y_pilot = Y[:, :pilot_len]
H_est = Y_pilot @ np.linalg.pinv(X_pilot)   # 最小二乘初始信道估计

# ========== Step 2: VB 迭代检测 ==========
# 初始化符号的均值和方差
mu_x = np.zeros((K, T - pilot_len), dtype=complex)
var_x = np.ones_like(mu_x) * 1.0

for iter in range(10):
    # Step 2.1: E-step - 估计每个数据符号的期望
    Y_data = Y[:, pilot_len:]
    res = Y_data - H_est @ mu_x
    for k in range(K):
        hk = H_est[:, k][:, None]
        interf = Y_data - H_est @ mu_x + hk @ mu_x[k:k+1, :]
        # 近似的后验均值更新（假设BPSK）
        num = np.real(np.conj(hk.T) @ Y_data)
        den = np.sum(np.abs(hk)**2) + noise_var
        mu_x[k, :] = np.tanh(num / den)
        var_x[k, :] = 1 - np.abs(mu_x[k, :])**2

    # Step 2.2: M-step - 更新信道估计
    X_est = np.concatenate([X_pilot, mu_x], axis=1)
    H_est = Y @ np.linalg.pinv(X_est)

# ========== Step 3: 检测结果 ==========
x_detect = np.sign(np.real(mu_x))
x_true_data = np.real(data)
ber = np.mean(x_detect != x_true_data)
print(f"BER = {ber:.3f}")

# ========== Step 4: 输出检查 ==========
print("\nTrue H:\n", np.round(H_true,2))
print("\nEstimated H:\n", np.round(H_est,2))