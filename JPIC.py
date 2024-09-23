import numpy as np
eps = np.finfo(float).eps;

class JPIC:
    # constants
    # PUL (pulse)
    PUL_BIORT   = 1;        # bi-orthogonal pulse
    PUL_RECTA   = 2;        # rectangular pulse
    # MOD (modulation)
    MOD_OFDM = 10;
    MOD_OTFS = 20;          # perfect csi
    MOD_OTFS_EM = 21;
    MOD_OTFS_SP = 22;
    # CE (channel estimation)
    CE_LS = 1;
    # SD (symbol detection)
    # SD - BSO
    # SD - BSO - mean
    # SD - BSO - mean - 1st iter
    SD_BSO_MEAN_CAL_INIT_MMSE   = 1;
    SD_BSO_MEAN_CAL_INIT_MRC    = 2;
    SD_BSO_MEAN_CAL_INIT_LS     = 3;
    # SD - BSO - mean - other iters
    SD_BSO_MEAN_CAL_MRC = 1;    
    SD_BSO_MEAN_CAL_LS  = 2;
    # SD - BSO - var
    SD_BSO_VAR_TYPE_APPRO = 1;
    SD_BSO_VAR_TYPE_ACCUR = 2;
    SD_BSO_VAR_CAL_MMSE   = 1;
    SD_BSO_VAR_CAL_MRC    = 2;
    SD_BSO_VAR_CAL_LS     = 3;
    # SD - BSE
    SD_BSE_ON   = 1;
    SD_BSE_OFF  = 0;
    # SD - DSC
    # SD - DSC - instantaneous square error
    SD_DSC_ISE_MMSE    = 1;
    SD_DSC_ISE_MRC     = 2;
    SD_DSC_ISE_LS      = 3;
    # SD - DSC - mean previous source
    SD_DSC_MEAN_PREV_SOUR_BSE = 1;
    SD_DSC_MEAN_PREV_SOUR_DSC = 2;
    # SD - DSC - variance previous source
    SD_DSC_VAR_PREV_SOUR_BSE = 1;
    SD_DSC_VAR_PREV_SOUR_DSC = 2;
    # SD - OUT
    SD_OUT_BSE = 1;
    SD_OUT_DSC = 2;
    
    # properties
    constel = None;
    constel_len = 0;
    es = 1;                                             # constellation average power
    # Pulse
    pulse_type              = PUL_RECTA;
    # MOD
    mod_type                = MOD_OTFS;
    # CE
    ce_type                 = CE_LS;
    # SD
    sd_bso_mean_cal_init    = SD_BSO_MEAN_CAL_INIT_MMSE;
    sd_bso_mean_cal         = SD_BSO_MEAN_CAL_MRC;
    sd_bso_var              = SD_BSO_VAR_TYPE_ACCUR;
    sd_bso_var_cal          = SD_BSO_VAR_CAL_MRC;
    sd_bse                  = SD_BSE_OFF;
    sd_dsc_ise              = SD_DSC_ISE_MRC;
    sd_dsc_mean_prev_sour   = SD_DSC_MEAN_PREV_SOUR_BSE;
    sd_dsc_var_prev_sour    = SD_DSC_VAR_PREV_SOUR_BSE;
    sd_out                  = SD_OUT_DSC;
    # other settings
    is_early_stop           = False;
    min_var                 = eps;      # the default minimal variance is 2.2204e-16
    iter_num                = 10;       # maximal iteration
    iter_diff_min           = eps;      # the minimal difference between 2 adjacent iterations 
    # OTFS configuration
    sig_len = 0;
    data_len = 0;
    M = 0;                              # the subcarrier number
    N = 0;                              # the timeslot number
    Xp = [];                            # pilots values (a matrix)
    XdLocs = [];                        # data locations matrix
    
    '''
    constructor
    @constellation:     the constellation, a vector.
    @min_var:           the minimal variance.
    @iter_num:          the maximal iteration.
    @iter_diff_min:     the minimal difference in **DSC** to early stop.
    '''
    def __init__(self, constel):
        constel = np.asarray(constel).squeeze();
        if constel.ndim != 1:
            raise Exception("The constellation must be a vector.");
        else:
            self.constel = constel;
            self.constel_len = len(constel);
            self.es = np.sum(abs(constel)**2)/self.constel_len;
        #TODO: add optionl inputs
        
    '''
    settings - pulse type
    '''
    def setPul2Biort(self):
        self.pulse_type = self.PUL_BIORT;
    def setPul2Recta(self):
        self.pulse_type = self.PUL_RECTA;
    
    '''
    settings - MOD - OFDM
    '''
    def setMod2Ofdm(self):
        self.mod_type = self.MOD_OFDM;
    '''
    settings - MOD - OTFS
    '''    
    def setMod2Otfs(self, M, N):
        self.mod_type = self.MOD_OTFS;
    