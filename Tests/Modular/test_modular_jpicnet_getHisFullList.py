import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from OTFSConfig import OTFSConfig
from JPICNet import JPICNet

print("------------------------------------------------------------------------")
print("JPICNet: getHisFullList()\n")
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
# channel configuration
p = 6;
lmax = 3;
kmax = 5;

'''
unbatched case
'''
print("unbatched case")
# init generalised variables
# config
otfsconfig = OTFSConfig();
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS();

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [N, M]);
# generate X_DD
X_DD = syms_mat
# generate
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
sigma2 = 1/np.power(10,SNR_d/10);

otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
HDD = otfs.getChannel();

# joint detection
jpicnet = JPICNet(constel);
# Tests - Recta - OTFS(Embed)
jpicnet.setPul2Recta();
jpicnet.setMod2OtfsEM(M, N);

his_new, his_mask = jpicnet.getHisFullList(his, lis, kis, lmax, kmax);
lis_full = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);
kis_full = np.tile(np.arange(-kmax, kmax+1), lmax+1);
# case 0
HDD_full = otfs.getChannel(his_new, lis_full, kis_full);
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new difference is %e"%HDD_diff_max);
# case 1
HDD_full = otfs.getChannel(his_new*his_mask, lis_full, kis_full);
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new*his_mask maximal difference is %e"%HDD_diff_max);

'''
batched case - 1
'''
print("batched case - 1")
# batch
batch_size = 1;
# init generalised variables
# config
otfsconfig = OTFSConfig(batch_size = batch_size);
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, N, M]);
# generate X_DD
X_DD = syms_mat
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
sigma2 = 1/np.power(10,SNR_d/10);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
HDD = otfs.getChannel();

# joint detection
jpicnet = JPICNet(constel, B=batch_size);
# Tests - Recta - OTFS(Embed)
jpicnet.setPul2Recta();
jpicnet.setMod2OtfsEM(M, N);

his_new, his_mask = jpicnet.getHisFullList(his, lis, kis, lmax, kmax);
lis_full = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);
kis_full = np.tile(np.arange(-kmax, kmax+1), lmax+1);
# case 0
HDD_full = otfs.getChannel(his_new, np.tile(lis_full, (batch_size, 1)), np.tile(kis_full, (batch_size, 1)));
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new difference is %e"%HDD_diff_max);
# case 1
HDD_full = otfs.getChannel(his_new*his_mask, np.tile(lis_full, (batch_size, 1)), np.tile(kis_full, (batch_size, 1)));
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new*his_mask maximal difference is %e"%HDD_diff_max);


'''
batched case - 10
'''
print("batched case - 10")
# batch
batch_size = 10;
# init generalised variables
# config
otfsconfig = OTFSConfig(batch_size = batch_size);
otfsconfig.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, M, N);
otfsconfig.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, N, M]);
# generate X_DD
X_DD = syms_mat
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
sigma2 = 1/np.power(10,SNR_d/10);
otfs.setChannel(p, lmax, kmax);
otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
HDD = otfs.getChannel();

# joint detection
jpicnet = JPICNet(constel, B=batch_size);
# Tests - Recta - OTFS(Embed)
jpicnet.setPul2Recta();
jpicnet.setMod2OtfsEM(M, N);

his_new, his_mask = jpicnet.getHisFullList(his, lis, kis, lmax, kmax);
lis_full = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);
kis_full = np.tile(np.arange(-kmax, kmax+1), lmax+1);
# case 0
HDD_full = otfs.getChannel(his_new, np.tile(lis_full, (batch_size, 1)), np.tile(kis_full, (batch_size, 1)));
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new difference is %e"%HDD_diff_max);
# case 1
HDD_full = otfs.getChannel(his_new*his_mask, np.tile(lis_full, (batch_size, 1)), np.tile(kis_full, (batch_size, 1)));
HDD_diff = abs(HDD_full - HDD);
HDD_diff_max = np.max(HDD_diff);
print(" - his_new*his_mask maximal difference is %e"%HDD_diff_max);
print("------------------------------------------------------------------------")