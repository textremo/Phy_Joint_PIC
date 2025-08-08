# from IPython import get_ipython
# #get_ipython().magic('reset -f')
# get_ipython().magic('clear')

import numpy as np
import torch
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid
from OTFSConfig import OTFSConfig
from CPE import CPE
from JPIC import JPIC
from Utils.utils import realH2Hfull

torch.set_default_dtype(torch.float64)


# configuration
K = 15;     # timeslote number
L = 16;     # subcarrier number
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

# init generalised variables
# config
oc = OTFSConfig();
oc.setFrame(OTFSConfig.FRAME_TYPE_GIVEN, K, L);
oc.setPul(OTFSConfig.PUL_TYPE_RECTA);
# OTFS module
otfs = OTFS(batch_size=batch_size);
# CPE
cpe = CPE(oc, lmax, kmax, Es_d, No, B=batch_size);
# pilots
X_p = cpe.genPilots(Es_p);

# Tx
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, K*L));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, K, L]);
# generate X_DD
X_DD = syms_mat + X_p
xDD = np.reshape(X_DD, [batch_size, K*L])
# generate
rg = OTFSResGrid(L, K, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();

# channel
otfs.modulate(rg);
otfs.setChannel(p, lmax, kmax)
otfs.passChannel(np.tile(No, batch_size))
his, lis, kis = otfs.getCSI()
Hdd = otfs.getChannel()

# Rx
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, K*L]);

# initial CHE
his_est, his_var, his_est_mask = cpe.estPaths(Y_DD, is_all=True)
kmin, kmax = cpe.getKRange()
# transfer data to real
#Y_DD = np.concatenate([np.real(Y_DD)[:, np.newaxis], np.imag(Y_DD)[:, np.newaxis]], 1)
#X_p = np.concatenate([np.real(X_p)[:, np.newaxis], np.imag(X_p)[:, np.newaxis]], 1)
#his_est = np.concatenate([np.real(his_est)[:, np.newaxis], np.imag(his_est)[:, np.newaxis]], 1)
#his_est_mask = np.repeat(his_est_mask[:, np.newaxis], 2, axis=1)

# joint detection
jpic = JPIC(oc, constel, lmax, kmin, kmax, iter_num=iter_num, B=batch_size);
jpic.setSdBsoMealCalInit2MMSE();
x, Hdd_est = jpic.detect(Y_DD, X_p, his_est, his_var, his_est_mask, No, sym_map=True);

