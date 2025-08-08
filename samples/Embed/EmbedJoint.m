% the reference for embed
clear;
clc;
EmbedConfig;

%% simulations
SERs_joint_jpic_mrc_mmse = zeros(length(SNR_ds), 1);
SERs_joint_jpic_mrc = zeros(length(SNR_ds), 1);
% simulate
for SNR_di = 1:length(SNR_ds)
    % other settings
    SNR_d = SNR_ds(SNR_di);
    No = Nos(SNR_di);
    N_fram = N_frams(SNR_di);
    pil_pow = 10^((SNR_p - SNR_d)/10);
    pil_thr = 3*sqrt(No);
    fprintf("SNR=%d\n",SNR_d);
    % frames
    tmp_SERs_joint_jpic_mrc_mmse = zeros(N_fram, 1);
    tmp_SERs_joint_jpic_mrc = zeros(N_fram, 1);
    parfor i_fram = 1:N_fram
        % generate data
        nbits = randi([0,1],sig_len*M_bits,1);
        xDD_syms = qammod(nbits, M_mod,'InputType','bit','UnitAveragePower',true);
        % data to rg
        rg = OTFSResGrid(M, N);
        rg.setPulse2Recta();
        rg.setPilot2Center(1, 1);
        rg.setGuard(gdn_len, gdp_len, gkn_len, gkp_len);
        rg.map(xDD_syms, "pilots_pow", pil_pow);
        XdLocs = rg.getContentDataLocsMat();
        Xp = rg.getPilotsMat();
        % pass the channel
        otfs = OTFS();
        otfs.modulate(rg);
        otfs.setChannel(p, lmax, kmax);
        otfs.passChannel(No);
        [csi_his, csi_lis, csi_kis] = otfs.getCSI();
        H_DD = otfs.getChannel();
        % Rx
        rg_rx = otfs.demodulate();
        [yDD, his, lis, kis] = rg_rx.demap("threshold", pil_thr);
        Y_DD = rg_rx.getContent();

        % joint
        % joint - mrc - mmse
        jpic_mrc_mmse = JPIC(constel);
        jpic_mrc_mmse.setPul2Recta();
        jpic_mrc_mmse.setMod2OtfsEM(M, N, "Xp", Xp, "XdLocs", XdLocs);
        [x_jpic_mrc_mmse, ~] = jpic_mrc_mmse.detect(Y_DD, lmax, kmax, No, "sym_map", true);
        tmp_SERs_joint_jpic_mrc_mmse(i_fram) = sum(abs(x_jpic_mrc_mmse - xDD_syms) > eps*sqrt(2))/sig_len;
        % joint - mrc
        jpic_mrc = JPIC(constel);
        jpic_mrc.setPul2Recta();
        jpic_mrc.setMod2OtfsEM(M, N, "Xp", Xp, "XdLocs", XdLocs);
        jpic_mrc.setSdBsoMealCalInit2MRC();
        [x_jpic_mrc, ~] = jpic_mrc.detect(Y_DD, lmax, kmax, No, "sym_map", true);
        tmp_SERs_joint_jpic_mrc(i_fram) = sum(abs(x_jpic_mrc - xDD_syms) > eps*sqrt(2))/sig_len;
    end
    % get the average
    SERs_joint_jpic_mrc_mmse(SNR_di) = mean(tmp_SERs_joint_jpic_mrc_mmse);
    SERs_joint_jpic_mrc(SNR_di) = mean(tmp_SERs_joint_jpic_mrc);
end

%% save
save(path_file_joint);

%% plot
figure;
% plot - threshold
semilogy(SNR_ds, max(SERs_joint_jpic_mrc_mmse, eps), "-->", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_joint_jpic_mrc, eps), "--s", LineWidth=2)
hold off;
grid on;
legend("JPIC-MMSE", "JPIC");
ylim([1e-6, 1e-1]);
ylabel("SER");
xlabel("SNR(Data)");
