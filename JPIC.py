import numpy as np
eps = np.finfo(float).eps;
from whatshow_toolbox import *;

class JPIC(MatlabFuncHelper):
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
        self.setOTFS(M, N);
    def setMod2OtfsEM(self, M, N, *, Xp=None, XdLocs=None):
        self.mod_type = self.MOD_OTFS_EM;
        self.setOTFS(M, N, Xp=Xp, XdLocs=XdLocs);
    def setMod2OtfsSP(self, M, N, *, Xp=None, XdLocs=None):
        self.mod_type = self.MOD_OTFS_SP;
        self.setOTFS(M, N, Xp=Xp, XdLocs=XdLocs);
    
    '''
    settings - CE
    '''
    def setCE2LS(self):
        self.ce_type = self.CE_LS;
    
    ########################################################
    # Auxiliary Methods
    '''
    build Phi - the channel estimation matrix
    @X:     the Tx matrix in DD domain ([batch_size], doppler, delay)
    @lmax:  the maximal delay
    @kmax:  the maximal Doppler
    '''
    def buildPhi(self, X, lmax, kmax):
        pmax = (lmax+1)*(2*kmax+1);                                         # the number of all possible paths
        lis = np.kron(np.arange(lmax+1), np.ones(2*kmax + 1)).astype(int);  # the delays on all possible paths
        kis = np.tile(np.arange(-kmax, kmax+1), lmax+1);                    # the dopplers on all possible paths
        Phi = self.zeros(self.sig_len, pmax).astype(complex);               # the return matrix
        for yk in range(self.N):
            for yl in range(self.M):
                Phi_ri = yk*self.M + yl;      # row id in Phi
                for p_id in range(pmax):
                    # path delay and doppler
                    li = lis[p_id];
                    ki = kis[p_id];
                    # x(k, l)
                    xl = yl - li;
                    if yl < li:
                        xl = xl + self.M;
                    xk = np.mod(yk - ki, self.N);
                    # exponential part (pss_beta)
                    if self.pulse_type == self.PUL_BIORT:
                        pss_beta = np.exp(-2j*np.pi*li*ki/self.M/self.N);
                    elif self.pulse_type == self.PUL_RECTA:
                        pss_beta = np.exp(2j*np.pi*(yl - li)*ki/self.M/self.N); # here, you must use `yl-li` instead of `xl` or there will be an error
                        if yl < li:
                            pss_beta = pss_beta*np.exp(-2j*np.pi*xk/self.N);
                    # assign value
                    Phi[..., Phi_ri, p_id] = X[..., xk, xl]*pss_beta;
        return Phi;
    
    ########################################################
    # private methods
    '''
    set OTFS
    @M:             subcarrier number
    @N:             timeslot numberr
    @Xp(opt):       the pilot value matrix (N, M)
    @XdLocs(opt):   the data locs matrix (N, M)
    '''
    def setOTFS(self, M, N, *, Xp=None, XdLocs=None):
        # OTFS size
        if M < 1 or not isinstance(M, int):
            raise Exception("The subcarrier number cannot be less than 1 or a fractional number");
        else:
            self.M = M;
        if N < 1 or not isinstance(N, int):
            raise Exception("The timeslot number cannot be less than 1 or a fractional number.");
        else:
            self.N = N;
        self.sig_len = M*N;
        self.data_len = M*N;
        # opt
        # opt - the pilot value matrix
        if Xp is not None:
            self.Xp = np.asarray(Xp);
            if self.Xp.shape[-2] != self.N:
                raise Exception("The timeslot number of the pilot matrix is not same as the given timeslot number.");
            elif self.Xp.shape[-1] != self.M:
                raise Exception("The subcarrier number of the pilot matrix is not same as the given subcarrier number.");
        # opt - the data locs matrix
        if XdLocs is not None:
            self.XdLocs = np.asarray(XdLocs);
            if self.XdLocs.shape[-2] != self.N:
                raise Exception("The timeslot number of the data location matrix is not same as the given timeslot number.");
            elif self.XdLocs.shape[-1] != self.M:
                raise Exception("The subcarrier number of the data location matrix is not same as the given subcarrier number.");
            self.data_len = np.sum(self.XdLocs);