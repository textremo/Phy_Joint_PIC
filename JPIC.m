classdef JPIC < handle
    % constants
    properties(Constant)
        % MOD (modulation)
        MOD_MIMO = 10;
        MOD_OTFS = 20;          % perfect csi
        MOD_OTFS_EM = 21;
        MOD_OTFS_SP = 22;
        % CE (channel estimation)
        CE_LS = 1;
        % SD (symbol detection)
        % SD - BSO
        % SD - BSO - mean
        % SD - BSO - mean - 1st iter
        SD_BSO_MEAN_CAL_INIT_MMSE   = 1;
        SD_BSO_MEAN_CAL_INIT_MRC    = 2;
        SD_BSO_MEAN_CAL_INIT_LS     = 3;
        % SD - BSO - mean - other iters
        SD_BSO_MEAN_CAL_MRC = 1;    
        SD_BSO_MEAN_CAL_LS  = 2;
        % SD - BSO - var
        SD_BSO_VAR_TYPE_APPRO = 1;
        SD_BSO_VAR_TYPE_ACCUR = 2;
        SD_BSO_VAR_CAL_MMSE   = 1;
        SD_BSO_VAR_CAL_MRC    = 2;
        SD_BSO_VAR_CAL_LS     = 3;
        % SD - BSE
        SD_BSE_ON   = 1;
        SD_BSE_OFF  = 0;
        % SD - DSC
        % SD - DSC - instantaneous square error
        SD_DSC_ISE_MRC     = 1;
        SD_DSC_ISE_ZF      = 2;
        SD_DSC_ISE_MMSE    = 3;
        % SD - DSC - mean previous source
        SD_DSC_MEAN_PREV_SOUR_BSE = 1;
        SD_DSC_MEAN_PREV_SOUR_DSC = 2;
        % SD - DSC - variance previous source
        SD_DSC_VAR_PREV_SOUR_BSE = 1;
        SD_DSC_VAR_PREV_SOUR_DSC = 2;
        % SD - OUT
        SD_OUT_BSE = 1;
        SD_OUT_DSC = 2;
    end
    % properties
    properties
        constel {mustBeNumeric}
        constel_len {mustBeNumeric}
        es = 1;                                             % constellation average power
        % MOD
        mod_type = JPIC.MOD_OTFS;
        % CE
        ce_type = JPIC.CE_LS;
        % SD
        sd_bso_mean_cal_init = JPIC.BSO_MEAN_INIT_MMSE;
        sd_bso_mean_cal = JPIC.BSO_MEAN_CAL_MRC;
        sd_bso_var = JPIC.SD_BSO_VAR_TYPE_ACCUR;
        sd_bso_var_cal = JPIC.BSO_VAR_CAL_MRC;
        sd_dsc_ise = JPIC.DSC_ISE_MRC;
        sd_dsc_mean_prev_sour = JPIC.DSC_MEAN_PREV_SOUR_BSE;
        sd_dsc_var_prev_sour = JPIC.DSC_VAR_PREV_SOUR_BSE;
        detect_sour = BPIC.DETECT_SOUR_DSC;
        % other settings
        is_early_stop = false;
        min_var {mustBeNumeric} = eps       % the default minimal variance is 2.2204e-16
        iter_num = 10                       % maximal iteration
        iter_diff_min = eps;                % the minimal difference between 2 adjacent iterations
    end
end