%
% Test JPIC
% - Ideal/Rectangular pulse
% - OTFS (Embed)
%
clear;
clc;

%% Data
M_mod = 4;
M_bits = log2(M_mod);
constel = qammod(0: M_mod-1, M_mod, 'UnitAveragePower',true);
SNR_p = 30; % dB
SNR_d = 18; % dB
No = 10^(-SNR_d/10);
pil_pow = 10^((SNR_p - SNR_d)/10);
sig_pow = 10^(SNR_d/10);
pil_thr = 3*sqrt(No);
M = 8;
N = 8;
kmax = 1;
lmax = 2;
p = 3;
pmax = (lmax + 1)*(2*kmax + 1);
% pilot
pk_len = 1;
pl_len = 1;
gkn_len = 2*kmax;
gkp_len = 2*kmax;
gln_len = lmax;
glp_len = lmax;
% Tx
xDD_idx = randi(4, M*N - (gkn_len + gkp_len + 1)*(gln_len + glp_len + 1), 1);
xDD = constel(xDD_idx);
%xDD = repmat(1+1j, M*N - (gkn_len + gkp_len + 1)*(gln_len + glp_len + 1), 1);
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Embed();
rg.setPilot2Center(pl_len, pk_len);
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
rg.map(xDD, 'pilots_pow', pil_pow);
X = rg.getContent();
XdLocs = rg.getContentDataLocsMat();
Xp = rg.getPilotsMat();
% channel - recta
otfs = OTFS();
otfs.setPulse2Recta();
his = [-0.245001294484229 + 0.190566910093579i	-0.0966735321794913 - 0.0195910978032867i	0.159574806792314 - 0.0563555055791835i	0.299959634256755 - 0.0628175424162719i	-0.322634497319953 - 0.0874996333904274i	0.363920822533598 + 0.00665269873273877i];
lis = [0,0,0,1,1,1];
kis = [-1,0,1,-1,0,1];
otfs.setChannel(his, lis, kis);
[his_acc, lis_acc, kis_acc] = otfs.getCSI();
otfs.modulate(rg);
otfs.passChannel(0);
Hdd_acc_recta = otfs.getChannel("data_only", false);
% channel - ideal
otfs.setPulse2Ideal();
Hdd_acc_ideal = otfs.getChannel("data_only", false);
% test channel
diff_channel = sum(abs(Hdd_acc_recta - Hdd_acc_ideal), "all");
assert(diff_channel > M*M*N*N*sqrt(2)*eps);

%% Tests
his_all = [his, zeros(1, pmax - length(his))].';
jpic = JPIC(constel);
jpic.setMod2OtfsEM(M, N, "Xp", Xp, "XdLocs", XdLocs);
% test - ideal pulse
jpic.setPul2Biort();
Hdd_ideal = jpic.buildHdd(his_all, lmax, kmax);
diff_Hdd_ideal = sum(abs(Hdd_acc_ideal - Hdd_ideal), "all");
assert(diff_Hdd_ideal == 0);
% test - recta pulse
jpic.setPul2Recta();
Hdd_recta = jpic.buildHdd(his_all, lmax, kmax);
diff_Hdd_recta = sum(abs(Hdd_acc_recta - Hdd_recta), "all");
assert(diff_Hdd_recta == 0);
disp("JPIC:buildHdd() test pass!");