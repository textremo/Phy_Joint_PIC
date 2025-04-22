import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from OTFSConfig import OTFSConfig
from CPE import CPE

# configuration
N = 15; # timeslote number
M = 16;  # subcarrier number
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
# config
otfsconfig = OTFSConfig(batch_size = batch_size);
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_IDEAL);
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
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);

'''
channel
'''
otfs.modulate(rg);

sigma2 = 1/np.power(10,SNR_d/10);

otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();


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

yDD_diff = abs(yDD_est - yDD);
yDD_diff_sum = np.sum(yDD_diff);
yDD_diff_max = np.max(yDD_diff);
Y_DD_power_diff = abs(Y_DD)**2;
X_DD_pow = abs(X_DD)**2;

