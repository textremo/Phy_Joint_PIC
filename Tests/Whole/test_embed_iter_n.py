import multiprocessing
import numpy as np
import sys
import scipy.io
import os
sys.path.append("../../..");
from JPIC import JPIC
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
eps = np.finfo(float).eps;
batch_size = multiprocessing.cpu_count()*4;    # batch size

# load the config
project_name = "phy_joint_pic";  # file
path_folder = os.path.abspath(os.path.dirname(__file__)).lower();
path_folder = path_folder[:path_folder.find(project_name)+len(project_name)];
path_file = os.path.normpath(path_folder+"/_data/Samples/Whole_CE/test_embedded_iter_n.mat");
# load matlab data
try:
    matlab_data = scipy.io.loadmat(path_file);
except FileNotFoundError:
    raise Exception("You have to run matlab script to generate data.");
# data
iter_num = np.squeeze(matlab_data['iter_num']);
constel = np.squeeze(matlab_data['constel']);
SNR_p = np.squeeze(matlab_data['SNR_p']).tolist(); # dB
SNR_d = np.squeeze(matlab_data['SNR_d']).tolist(); # dB
No = np.squeeze(matlab_data['No']).tolist();
pil_pow = np.squeeze(matlab_data['pil_pow']).tolist();
sig_pow = np.squeeze(matlab_data['sig_pow']).tolist();
pil_thr = np.squeeze(matlab_data['pil_thr']).tolist();
M = np.squeeze(matlab_data['M']).tolist();
N = np.squeeze(matlab_data['N']).tolist();
p = np.squeeze(matlab_data['p']).tolist();
kmax = np.squeeze(matlab_data['kmax']).tolist();
lmax = np.squeeze(matlab_data['lmax']).tolist();
# pilot
pk_len = np.squeeze(matlab_data['pk_len']).tolist();
pl_len = np.squeeze(matlab_data['pl_len']).tolist();
gkn_len = np.squeeze(matlab_data['gkn_len']).tolist();
gkp_len = np.squeeze(matlab_data['gkp_len']).tolist();
gln_len = np.squeeze(matlab_data['gln_len']).tolist();
glp_len = np.squeeze(matlab_data['glp_len']).tolist();
# CSI
his = np.squeeze(matlab_data['his']);
lis = np.squeeze(matlab_data['lis']);
kis = np.squeeze(matlab_data['kis']);
noise = np.squeeze(matlab_data['noise']);
# data for comparison
Y_DD_mat = np.squeeze(matlab_data['Y_DD']);
H_DD_mat = np.squeeze(matlab_data['H_DD']);

# Tx
xDD = np.squeeze(matlab_data['xDD']);
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Embed();
rg.setPilot2Center(pl_len, pk_len);
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
rg.map(xDD, pilots_pow=pil_pow);
XdLocs = rg.getContentDataLocsMat();
Xp = rg.getPilotsMat();
# channel - ideal
otfs = OTFS();
otfs.setPulse2Recta();
otfs.setChannel(his, lis, kis);
[his_acc, lis_acc, kis_acc] = otfs.getCSI();
otfs.modulate(rg);
otfs.passChannel(noise);
# Rx
rg_rx = otfs.demodulate();
[y, hiEsts, liEsts, kiEsts] = rg_rx.demap(threshold=pil_thr);
Y_DD = rg_rx.getContent();
assert(np.sum(abs(Y_DD_mat-Y_DD))<N*M*eps);

# Test
jpic = JPIC(constel, iter_num=iter_num);
# Tests - Recta - OTFS(Embed)
jpic.setPul2Recta();
jpic.setMod2OtfsEM(M, N, Xp=Xp, XdLocs=XdLocs);
x, H_DD = jpic.detect(Y_DD, lmax, kmax, No, sym_map=True);
diff_x = abs(x - xDD);
assert(np.sum(abs(diff_x)) == 0);
diff_H_DD = abs(H_DD - H_DD_mat);
assert(np.sum(abs(diff_H_DD)) < eps*M*N*M*N);
