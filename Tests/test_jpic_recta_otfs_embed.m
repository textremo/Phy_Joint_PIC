%
% Test JPIC
% - Rectangular pulse
% - OTFS (Embed)
%
clear;
clc;

%% Data
M_mod = 16;
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
p = 3;
kmax = 1;
lmax = 2;
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
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.setPilot2Embed();
rg.setPilot2Center(pl_len, pk_len);
rg.setGuard(gln_len, glp_len, gkn_len, gkp_len);
rg.map(xDD, 'pilots_pow', pil_pow);
XdLocs = rg.getContentDataLocsMat();
Xp = rg.getPilotsMat();
% channel - ideal
otfs = OTFS();
otfs.setPulse2Recta();
otfs.setChannel(p, lmax, kmax);
[his_acc, lis_acc, kis_acc] = otfs.getCSI();
otfs.modulate(rg);
otfs.passChannel(No);
% Rx
rg_rx = otfs.demodulate();
[y, his, lis, kis] = rg_rx.demap("threshold", pil_thr);
Y_DD = rg_rx.getContent();

%% Tests
jpic = JPIC(constel);

%% Tests - Recta - OTFS(Embed)
jpic.setPul2Recta();
jpic.setMod2OtfsEM(M, N, "Xp", Xp, "XdLocs", XdLocs);
[x, H_DD] = jpic.detect(Y_DD, No, "sym_map", true);
