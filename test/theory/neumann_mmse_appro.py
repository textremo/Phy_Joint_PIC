import numpy as np
import torch
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector

N = 15;     # timeslote number
M = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 37;
No = 10**(-SNR_d/10);
Es_d = 1;
Es_p = 10**((SNR_p - SNR_d)/10);
# batch
batch_size = 10;
# channel configuration
p = 6;
lmax = 3;
kmax = 5;



'''
init generalised variables
'''
# OTFS module
otfs = OTFS(batch_size=batch_size);

'''
Tx
'''
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, N, M]);
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(syms_mat);
otfs.modulate(rg);

'''
channel
'''
No = 1/np.power(10,SNR_d/10);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(No);
H_DD = otfs.getChannel();

'''
Rx
'''
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, M*N, 1]);
print("- Rx");
# test y_DD diff
yDD_diff_abs = abs(yDD - H_DD @ np.expand_dims(syms_vec, axis=-1));
yDD_diff_abs_max = np.max(yDD_diff_abs);
print("  - max(|yDD - H_DD@xDD|) is %e"%yDD_diff_abs_max);

'''
symbol detection
'''
print("- symbol detection");
print("  - MMSE: ", end="");
H_DD_t = np.moveaxis(H_DD, -1, -2);
norm_inv =  np.linalg.inv(H_DD_t@H_DD +  np.tile(np.eye(N*M), [batch_size, 1,1]));
xDD_est_mmse = norm_inv @H_DD_t@yDD;
xDD_est_mmse = xDD_est_mmse.squeeze(-1);
xDD_est_mmse_diff_abs = abs(xDD_est_mmse - syms_vec);
xDD_est_mmse_diff_abs_max = np.max(xDD_est_mmse_diff_abs);
print("max(|xDD - xDD_est_mmse|) is %e"%xDD_est_mmse_diff_abs_max);
K = 3;
neumann_inv = np.zeros([batch_size, M*N, N*M], dtype=complex);
for k in range(K):
    neumann_inv += (-1)**k/No**(k-1)*np.linalg.matrix_power(H_DD_t@H_DD, k);
xDD_est_neumann_mmse = neumann_inv @ H_DD_t@yDD;
xDD_est_neumann_mmse = xDD_est_neumann_mmse.squeeze(-1);
xDD_est_neumann_mmse_diff0 = abs(xDD_est_neumann_mmse - xDD_est_mmse);
print("  - Neumann MMSE: ", end="");
