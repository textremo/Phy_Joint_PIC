clear;
clc;
EmbedConfig;
%% load
if ~exist(path_file_comp, "file")
    error("Please run `EmbedComp.m` to generate comparison data.");
else
    load(path_file_comp);
end
if ~exist(path_file_joint, "file")
    error("Please run `EmbedJoint.m` to generate comparison data.");
else
    load(path_file_joint);
end



%% plot
figure;
% plot - threshold
semilogy(SNR_ds, max(SERs_em_thresh_mmse, eps), "-->", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_thresh_bpic, eps), "--s", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_thresh_ep, eps), "--^", LineWidth=2)
hold on;
% plot - joint
semilogy(SNR_ds, max(SERs_joint_jpic_mrc_mmse, eps), "--<", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_joint_jpic_mrc, eps), "--", LineWidth=2)
% plot - perfect csi
semilogy(SNR_ds, max(SERs_em_percsi_mmse, eps), "->", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_percsi_bpic, eps), "-s", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_percsi_ep, eps), "-^", LineWidth=2)
hold off;
grid on;
legend("Thresh-MMSE", "Thresh-BPIC", "Thresh-EP", "JPIC-MMSE", "JPIC", "PerCSI-MMSE", "PerCSI-BPIC", "PerCSI-EP");
ylim([1e-6, 1e-1]);
ylabel("SER");
xlabel("SNR(Data)");