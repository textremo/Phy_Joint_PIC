#
# Test JPIC
# - Ideal/Rectangular pulse
# - OTFS (Embed)
import numpy as np
from whatshow_phy_mod_otfs import OTFS, OTFSResGrid, OTFSDetector
from JPIC import JPIC
eps = np.finfo(float).eps

'''
Data
'''
constel = [-0.707106781186548+0.707106781186548j, -0.707106781186548-0.707106781186548j, 0.707106781186548+0.707106781186548j, 0.707106781186548-0.707106781186548j];
SNR_p = 30; # dB
SNR_d = 18; # dB
No = 10**(-SNR_d/10);
pil_pow = 10**((SNR_p - SNR_d)/10);
sig_pow = 10**(SNR_d/10);
pil_thr = 3*np.sqrt(No);
M = 8;
N = 8;
kmax = 1;
lmax = 2;
p = 3;
pmax = (lmax + 1)*(2*kmax + 1);
# pilot
pk_len = 1;
pl_len = 1;
gkn_len = 2*kmax;
gkp_len = 2*kmax;
gln_len = lmax;
glp_len = lmax;
# Tx
xDD_idx = np.random.randint(4, size=M*N-(gkn_len + gkp_len + 1)*(gln_len + glp_len + 1));
xDD = np.take(constel, xDD_idx);
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Embed();
rg.setPilot2Center(pl_len, pk_len);
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
rg.map(xDD, pilots_pow=pil_pow);
X = rg.getContent();
XdLocs = rg.getContentDataLocsMat();
Xp = rg.getPilotsMat();
# channel - recta
otfs = OTFS();
otfs.setPulse2Recta();
his = [-0.245001294484229+0.190566910093579j, -0.0966735321794913-0.0195910978032867j, 0.159574806792314-0.0563555055791835j, 0.299959634256755-0.0628175424162719j, -0.322634497319953-0.0874996333904274j, 0.363920822533598+0.00665269873273877j];
lis = [0,0,0,1,1,1];
kis = [-1,0,1,-1,0,1];
otfs.setChannel(his, lis, kis);
[his_acc, lis_acc, kis_acc] = otfs.getCSI();
otfs.modulate(rg);
otfs.passChannel(0);
Hdd_acc_recta = otfs.getChannel(data_only=False);
# channel - ideal
otfs.setPulse2Ideal();
Hdd_acc_ideal = otfs.getChannel(data_only=False);
# test channel
diff_channel = np.sum(abs(Hdd_acc_recta - Hdd_acc_ideal));
assert(diff_channel > M*M*N*N*np.sqrt(2)*eps);

'''
Tests
'''
his_all = np.concatenate([his, np.zeros(pmax - len(his))], axis=-1);
jpic = JPIC(constel);
jpic.setMod2OtfsEM(M, N, Xp=Xp, XdLocs=XdLocs);
# test - ideal pulse
jpic.setPul2Biort();
Hdd_ideal = jpic.buildHdd(his_all, lmax, kmax);
diff_Hdd_ideal = np.sum(abs(Hdd_acc_ideal - Hdd_ideal));
assert(diff_Hdd_ideal == 0);
# test - recta pulse
jpic.setPul2Recta();
Hdd_recta = jpic.buildHdd(his_all, lmax, kmax);
diff_Hdd_recta = np.sum(abs(Hdd_acc_recta - Hdd_recta));
assert(diff_Hdd_recta == 0);
print("JPIC:buildHdd() test pass!");