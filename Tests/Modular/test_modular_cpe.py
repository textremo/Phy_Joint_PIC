from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')

import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from OTFSConfig import OTFSConfig
from CPE import CPE
from JPICNet import JPICNet
from Utils.utils import realH2Hfull

print("------------------------------------------------------------------------")
print("CPE\n")


# configuration
N = 15;     # timeslote number
M = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = 14;
SNR_p = 40.25399848;
No = 10**(-SNR_d/10);
Es_d = 1;
Es_p = 10**((SNR_p - SNR_d)/10);
# batch
batch_size = 10;
# channel configuration
p = 6;
lmax = 3;
kmax = 5;

# JPIC config
iter_num = 10;

'''
init generalised variables
'''
# config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);
# CPE
cpe = CPE(otfsconfig, lmax, kmax, Es_d, No, B=batch_size);
# pilots
X_p = cpe.genPilots(Es_p);


'''
Tx
'''
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, N, M]);
# generate X_DD
X_DD = X_p;
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

'''
channel
'''
otfs.modulate(rg);

sigma2 = 1/np.power(10,SNR_d/10);

# his = np.asarray([[0.340993+0.124189j, -0.388739+0.410122j, -0.0250011+0.596597j,-0.846615+0.157308j,-0.0281749+0.0820766j,-0.304782-0.0582677j],[0.0373028-0.251298j,0.119883+0.481477j,-0.135965-0.772444j,0.0369038-0.32628j,-0.0339525+0.168356j,-0.24526-0.575117j]]);
# kis = np.asarray([[2,2,-3,-5,-3,0],[-3,-3,1,4,5,4]]);
# lis = np.asarray([[0,2,2,2,3,1],[3,2,2,0,3,2]]);

otfs.setChannel(p, lmax, kmax);
otfs.passChannel(No);
#otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = realH2Hfull(kmax, lmax, his, lis, kis, batch_size=batch_size);
his_mask = abs(his_full) > 0

'''
Rx
'''
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, M*N]);


'''
signal processing
'''
xDD = np.reshape(X_DD, [batch_size, M*N]);
yDD_est = np.squeeze(H_DD @ np.expand_dims(xDD, axis=-1), axis=-1);
#yDD_diff = abs(yDD_est - yDD);
#yDD_diff_sum = np.sum(yDD_diff);
#yDD_diff_max = np.max(yDD_diff);
#Y_DD_power_diff = abs(Y_DD)**2;
#X_DD_pow = abs(X_DD)**2;

# estimate the paths
his_est, lis_est, kis_est = cpe.estPaths(Y_DD);
his_full_est = realH2Hfull(kmax, lmax, his_est, lis_est, kis_est, batch_size=batch_size);
his_full_diff = abs(his_full_est - his_full);


print("- CPE threshold (power): %f"%cpe.thres)
print("- CHE check (missing):")
diff_num_miss = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_mask[bid, his_shift] and y_pow <= cpe.thres:
                diff_num_miss += 1
                print(f"  - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}")
print("- CHE check (error):")
diff_num_erro = 0;
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if not his_mask[bid, his_shift] and y_pow > cpe.thres:
                diff_num_erro += 1
                print(f"  - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}, diff: {his_full_diff[bid, his_shift]: .4f}")
print("- CHE check (correct):")
diff_num_corr = 0;  
for bid in range(batch_size):
    for li in range(lmax + 1):
        for ki in range(-kmax, kmax+1):
            his_shift = li*(2*kmax+1) + kmax + ki;
            y_pow = abs(his_full_est[bid, his_shift]*cpe.pil_val)**2;
            if his_mask[bid, his_shift] and y_pow > cpe.thres:
                diff_num_corr += 1
                print(f"  - [{bid:2d}, {his_shift:2d}], origin: {his_full[bid, his_shift]:+.4f}, est: {his_full_est[bid, his_shift]:+.4f}, diff: {his_full_diff[bid, his_shift]: .4f}")


# if diff_num_less + diff_num_grea != np.sum(his_full_diff > 1e-13):
#     raise Exception("Difference not match!");
# else:
max_pos = np.unravel_index(np.argmax(his_full_diff), his_full_diff.shape)
print(f"- find {diff_num_miss + diff_num_erro} difference, at max [{max_pos[0]}, {max_pos[1]}] {np.max(his_full_diff):.4f}.")

print("------------------------------------------------------------------------")