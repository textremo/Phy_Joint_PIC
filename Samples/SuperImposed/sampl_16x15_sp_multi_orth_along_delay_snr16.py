import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from OTFSConfig import OTFSConfig
from CPE import CPE
from JPICNet import JPICNet
from Utils.utils import realH2Hfull

# configuration
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

# JPIC config
iter_num = 10;

'''
init generalised variables
'''
# config
otfsconfig = OTFSConfig(batch_size = batch_size);
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);
# CPE
cpe = CPE(otfsconfig, lmax, Es_d, No);
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
X_DD = syms_mat + X_p;
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

otfs.setChannel(p, lmax, kmax, isSameDD=True);
#otfs.setChannel(his, lis, kis, isSameDD=True);
otfs.passChannel(np.tile(No, batch_size));
otfs.passChannel(0);
#otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();
his_full = realH2Hfull(kmax, lmax, his, lis, kis, batch_size=batch_size);


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
his_est_full = realH2Hfull(kmax, lmax, his_est, lis_est, kis_est, batch_size=batch_size);
H_DD_est = otfs.getChannel(his_est, lis_est, kis_est);
H_DD_est_diff = abs(H_DD_est - H_DD);
H_DD_est_diff[H_DD_est_diff< 1e-10] = 0;
his_est_full_diff = abs(his_est_full - his_full);
print(np.max(his_est_full_diff))

a_flat = his_est_full_diff.reshape(batch_size, -1)
max_vals = np.max(a_flat, axis=1)
max_idxs = np.argmax(a_flat, axis=1)
positions = [np.unravel_index(idx, his_est_full_diff.shape[1:]) for idx in max_idxs]


'''
joint detection
'''
jpic = JPICNet(constel, iter_num=iter_num, batch_size=batch_size);

# Tests - Recta - OTFS(Embed)
jpic.setPul2Recta();
jpic.setMod2OtfsEM(M, N, Xp=X_p);
# x, H_DD = jpic.detect(Y_DD, lmax, kmax, No, sym_map=True);
# diff_x = abs(x - xDD);
