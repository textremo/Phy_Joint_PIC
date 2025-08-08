% the reference for embed
clear;
clc;
EmbedConfig;

%% simulations
SERs_em_percsi_mmse = zeros(length(SNR_ds), 1);
SERs_em_percsi_bpic = zeros(length(SNR_ds), 1);
SERs_em_percsi_ep = zeros(length(SNR_ds), 1);
SERs_em_thresh_mmse = zeros(length(SNR_ds), 1);
SERs_em_thresh_bpic = zeros(length(SNR_ds), 1);
SERs_em_thresh_ep = zeros(length(SNR_ds), 1);
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
    tmp_SERs_em_percsi_mmse = zeros(N_fram, 1);
    tmp_SERs_em_percsi_bpic = zeros(N_fram, 1);
    tmp_SERs_em_percsi_ep = zeros(N_fram, 1);
    tmp_SERs_em_thresh_mmse = ones(N_fram, 1);
    tmp_SERs_em_thresh_bpic = ones(N_fram, 1);
    tmp_SERs_em_thresh_ep = ones(N_fram, 1);
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

        % detect
        bpic = BPIC(constel, "detect_sour", BPIC.DETECT_SOUR_BSE);
        ep = EP(constel, "beta", 0.1, "epsilon", 1e-13);
        % detect - perfect CSI
        csi_mmse_xe = ep.symmap(inv(H_DD'*H_DD+ No*eye(M*N - (4*kmax+1)*(2*lmax+1)))*H_DD'*yDD);
        csi_bpic_xe = bpic.detect(yDD, H_DD, No, "sym_map", true);
        csi_ep_xe = ep.detect(yDD, H_DD, No, "sym_map", true);
        tmp_SERs_em_percsi_mmse(i_fram) = sum(abs(csi_mmse_xe - xDD_syms) > eps*sqrt(2))/sig_len;
        tmp_SERs_em_percsi_bpic(i_fram) = sum(abs(csi_bpic_xe - xDD_syms) > eps*sqrt(2))/sig_len;
        tmp_SERs_em_percsi_ep(i_fram) = sum(abs(csi_ep_xe - xDD_syms) > eps*sqrt(2))/sig_len;
        % detect - threshold
        if ~isempty(his)
            H_DDe = otfs.getChannel(his, lis, kis);
            mmse_xe = ep.symmap(inv(H_DDe'*H_DDe+ No*eye(M*N - (4*kmax+1)*(2*lmax+1)))*H_DDe'*yDD);
            bpic_xe = bpic.detect(yDD, H_DDe, No, "sym_map", true);
            ep_xe = ep.detect(yDD, H_DDe, No, "sym_map", true);
            tmp_SERs_em_thresh_mmse(i_fram) = sum(abs(mmse_xe - xDD_syms) > eps*sqrt(2))/sig_len;
            tmp_SERs_em_thresh_bpic(i_fram) = sum(abs(bpic_xe - xDD_syms) > eps*sqrt(2))/sig_len;
            tmp_SERs_em_thresh_ep(i_fram) = sum(abs(ep_xe - xDD_syms) > eps*sqrt(2))/sig_len;
        end
    end
    % get the average
    SERs_em_percsi_mmse(SNR_di) = mean(tmp_SERs_em_percsi_mmse);
    SERs_em_percsi_bpic(SNR_di) = mean(tmp_SERs_em_percsi_bpic);
    SERs_em_percsi_ep(SNR_di) = mean(tmp_SERs_em_percsi_ep);
    SERs_em_thresh_mmse(SNR_di) = mean(tmp_SERs_em_thresh_mmse);
    SERs_em_thresh_bpic(SNR_di) = mean(tmp_SERs_em_thresh_bpic);
    SERs_em_thresh_ep(SNR_di) = mean(tmp_SERs_em_thresh_ep);
end

%% save
save(path_file_comp);

%% plot
figure;
% plot - threshold
semilogy(SNR_ds, max(SERs_em_thresh_mmse, eps), "-->", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_thresh_bpic, eps), "--s", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_thresh_ep, eps), "--^", LineWidth=2)
hold on;
% plot - perfect csi
semilogy(SNR_ds, max(SERs_em_percsi_mmse, eps), "->", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_percsi_bpic, eps), "-s", LineWidth=2)
hold on;
semilogy(SNR_ds, max(SERs_em_percsi_ep, eps), "-^", LineWidth=2)
hold off;
grid on;
legend("Thresh-MMSE", "Thresh-BPIC", "Thresh-EP", "PerCSI-MMSE", "PerCSI-BPIC", "PerCSI-EP");
ylim([1e-6, 1e-1]);
ylabel("SER");
xlabel("SNR(Data)");