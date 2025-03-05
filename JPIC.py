import numpy as np
from numpy import exp
from numpy.linalg import inv
from numpy.linalg import norm as vecnorm
from whatshow_toolbox import *;
eps = np.finfo(float).eps;

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
    CE_MRC = 1;
    CE_ZF = 1;
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
    ce_type                 = CE_MRC;
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
    @constel:           the constellation, a vector.
    @min_var:           the minimal variance.
    @iter_num:          the maximal iteration.
    @iter_diff_min:     the minimal difference in **DSC** to early stop.
    '''
    def __init__(self, constel, *, min_var=None, iter_num=None, iter_diff_min = None, batch_size=None):
        constel = np.asarray(constel).squeeze();
        if constel.ndim != 1:
            raise Exception("The constellation must be a vector.");
        else:
            self.constel = constel;
            self.constel_len = len(constel);
            self.es = np.sum(abs(constel)**2)/self.constel_len;
        
        # optionl inputs
        if min_var is not None:
            self.min_var = min_var;
        if iter_num is not None:
            self.iter_num = iter_num;
        if iter_diff_min:
            self.iter_diff_min = iter_diff_min;
        # optionl inputs - batch_size
        if batch_size is not None:
            self.batch_size = batch_size;
        
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
    def setCE2MRC(self):
        self.ce_type = self.CE_MRC;
    def setCE2ZF(self):
        self.ce_type = self.CE_ZF;
        
    '''
    settings - SD
    '''
    # settings - SD - BSO - mean - cal (init)
    def setSdBsoMealCalInit2MMSE(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MMSE;
    def setSdBsoMealCalInit2MRC(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MRC;
    def setSdBsoMealCalInit2LS(self):
        self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_LS;
    # settings - SD - BSO - mean - cal
    def setSdBsoMeanCal2MRC(self):
        self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_MRC;    
    def setSdBsoMeanCal2LS(self):
        self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_LS;
    # settings - SD - BSO - var
    def setSdBsoVar2Appro(self):
        self.sd_bso_var = self.SD_BSO_VAR_TYPE_APPRO;
    def setSdBsoVar2Accur(self):
        self.sd_bso_var = self.SD_BSO_VAR_TYPE_ACCUR;
    # settings - SD - BSO - var - cal
    def setSdBsoVarCal2MMSE(self):
        self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MMSE;
    def setSdBsoVarCal2MRC(self):
        self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MRC;
    def setSdBsoVarCal2LS(self):
        self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_LS;
    # settings - SD - BSE
    def setSdBseOn(self):
        self.sd_bse = self.SD_BSE_ON;
    def setSdBseOff(self):
        self.sd_bse = self.SD_BSE_OFF;
    # settings - SD - DSC - instantaneous square error
    def setSdDscIse2MMSE(self):
        self.sd_dsc_ise = self.SD_DSC_ISE_MMSE;
    def setSdDscIse2MRC(self):
        self.sd_dsc_ise = self.SD_DSC_ISE_MRC;
    def setSdDscIse2LS(self):
        self.sd_dsc_ise = self.SD_DSC_ISE_LS;
    # settings - SD - DSC - mean previous source
    def setSdDscMeanPrevSour2BSE(self):
        self.sd_dsc_mean_prev_sour = self.SD_DSC_MEAN_PREV_SOUR_BSE;
    def setSdDscMeanPrevSour2DSC(self):
        self.sd_dsc_mean_prev_sour = self.SD_DSC_MEAN_PREV_SOUR_DSC;
    # settings - SD - DSC - variance previous source
    def setSdDscVarPrevSour2BSE(self):
        self.sd_dsc_var_prev_sour = self.SD_DSC_VAR_PREV_SOUR_BSE;
    def setSdDscVarPrevSour2DSC(self):
        self.sd_dsc_var_prev_sour = self.SD_DSC_VAR_PREV_SOUR_DSC;
    # settings - SD - OUT
    def setSdOut2BSE(self):
        if self.sd_bse == self.SD_BSE_OFF:
            raise Exception("Cannot use BSE output because BSE module is off.");
        self.sd_out = self.SD_OUT_BSE;
    def setSdOut2DSC(self):
        self.sd_out = self.SD_OUT_DSC;
    
    '''
    symbol detection
    @Y_DD:          the received signal in the delay Doppler domain [(batch_size), doppler, delay]
    @lmax:          the maximal delay index
    @kmax:          the maximal Doppler index
    @No:            the noise (linear) power
    @sym_map(opt):  false by default. If true, the output will be mapped to the constellation
    '''
    def detect(self, Y_DD, lmax, kmax, No, *, sym_map=False):
        Y_DD = np.asarray(Y_DD);
        No = np.asarray(No);
        
        # input check
        Y_DD_N = Y_DD.shape[-2];
        Y_DD_M = Y_DD.shape[-1];
        if Y_DD_N != self.N or Y_DD_M != self.M:
            raise Exception("The received matrix is not in the shape of (%d, %d)."%(self.N, self.M));
        if No.ndim > 0:
            raise Exception("The noise power is not a scalar.");
            
        # constant values
        y = self.reshape(Y_DD, self.sig_len, 1);
        xp = self.reshape(self.Xp, self.sig_len, 1);
        xdlocs = self.reshape(self.XdLocs, self.sig_len, 1);
        xndlocs =  self.reshape(np.invert(xdlocs), self.sig_len, 1);
        
        # iterative detection
        ise_dsc_prev = self.zeros(self.sig_len, 1);
        #x_bse = self.zeros(self.sig_len, 1);
        x_dsc = self.zeros(self.sig_len, 1).astype(complex);
        x_det = self.zeros(self.sig_len, 1).astype(complex);     # the soft symbol detection results
        for iter_id in range(self.iter_num):
            # CE
            Xe = self.reshape(x_det, self.N, self.M);
            Phi = self.buildPhi(Xe + self.Xp, lmax, kmax);
            Phit = np.conj(np.moveaxis(Phi, -1, -2));
            h = np.squeeze(inv(Phit @ Phi) @ Phit @ y, axis=-1);
            # build the channel
            H = self.buildHdd(h, lmax, kmax);
            Ht = np.conj(np.moveaxis(H, -1, -2));
            Hty = Ht @ y;
            HtH = Ht @ H;
            HtH_off = ((self.eye(self.sig_len)+1) - self.eye(self.sig_len)*2)*HtH;
            #HtH_off_sqr = np.square(HtH_off);
            # SD
            # SD - BSO
            # SD - BSO - mean
            if iter_id == 0:
                # SD - BSO - mean - 1st iter
                if self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MMSE:
                    bso_zigma_1 = inv(HtH + No/self.es*self.eye(self.sig_len));
                elif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_MRC:
                    bso_zigma_1 = self.diag(1/vecnorm(H, axis=-1)**2);
                elif self.sd_bso_mean_cal_init == self.SD_BSO_MEAN_CAL_INIT_LS:
                    bso_zigma_1 = inv(HtH);
                else:
                    bso_zigma_1 = self.eye(self.sig_len);
                x_bso = bso_zigma_1 @ (Hty - HtH_off @ x_dsc - HtH @ xp);
            else:
                # SD - BSO - mean - other iteration
                if self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_MRC:
                    bso_zigma_n = self.diag(1/vecnorm(H, axis=-1)**2);
                elif self.sd_bso_mean_cal == self.SD_BSO_MEAN_CAL_LS:
                    bso_zigma_n = inv(HtH);
                x_bso = bso_zigma_n @ (Hty - HtH_off@x_dsc - HtH@xp);
            x_bso[xndlocs] = 0;

            # SD - DSC
            if self.sd_dsc_ise == self.SD_DSC_ISE_MMSE:
                dsc_w = inv(HtH + No/self.es*self.eye(self.sig_len));
            elif self.sd_dsc_ise == self.SD_DSC_ISE_MRC:
                dsc_w = self.diag(1/vecnorm(H, axis=-1)**2);
            elif self.sd_dsc_ise ==  self.SD_DSC_ISE_LS:
                dsc_w = inv(HtH);
            ise_dsc = (dsc_w @ (Hty - HtH@(x_bso + xp)))**2;
            ies_dsc_sum = ise_dsc + ise_dsc_prev;
            ies_dsc_sum = self.max(ies_dsc_sum, self.min_var);
            # DSC - rho (if we use this rho, we will have a little difference)
            rho_dsc = ise_dsc_prev/ies_dsc_sum;
            # DSC - mean
            if iter_id == 0:
                x_dsc = x_bso;
            else:
                if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_BSE:
                    #x_dsc = ise_dsc./ies_dsc_sum.*x_bse_prev + ise_dsc_prev./ies_dsc_sum.*x_bse;
                    x_dsc = (1 - rho_dsc)*x_bso_prev + rho_dsc*x_bso;
                if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_DSC:
                    x_dsc = (1 - rho_dsc)*x_dsc + rho_dsc*x_bso;

            # update statistics
            # update statistics - BSE
            if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_BSE:
                x_bso_prev = x_bso;
            # update statistics - DSC - instantaneous square error
            ise_dsc_prev = ise_dsc;

            # soft symbol estimation
            x_det[xdlocs] = self.symmapNoBat(x_dsc[xdlocs]);
            x_det[xndlocs] = 0;
        # only keep data part
        x = x_det[xdlocs] if self.batch_size is self.BATCH_SIZE_NO else np.reshape(x_det[xdlocs], (self.batch_size, -1));
        return x, H;
    
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
    
    '''
    build Hdd
    @his: the path gains of all paths
    @lmax: the maximal delay
    @kmax: the maximal Doppler 
    @thres(opt): the threshold of a path (default 0)
    '''
    def buildHdd(self, his, lmax, kmax, *, thres=0):
        # channel parameters
        pmax = (lmax + 1)*(2*kmax+1);
        lis = np.kron(np.arange(0, lmax+1), np.ones(2*kmax+1)).astype(int);   
        kis = np.tile(np.arange(-kmax, kmax+1), lmax+1);
        # channel parameters - batch size
        if self.batch_size is not self.BATCH_SIZE_NO:
            lis = np.tile(lis, (self.batch_size, 1));
            kis = np.tile(kis, (self.batch_size, 1));
        # filter the path gain
        his = np.asarray(his);
        his[abs(his)<abs(thres)] = 0;
        # build the channel in the DD 
        Hdd = None;
        if self.pulse_type == self.PUL_BIORT:
            Hdd = self.buildOtfsBiortDDChannel(pmax, his, lis, kis);
        elif self.pulse_type == self.PUL_RECTA:
            Hdd = self.buildOtfsRectaDDChannel(pmax, his, lis, kis);
        else:
            raise Exception("The pulse type is not recognised.");
        return Hdd;
    
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
            self.Xp = np.asarray(Xp) if self.batch_size is self.BATCH_SIZE_NO else np.tile(Xp, (self.batch_size, 1, 1));
            if self.Xp.shape[-2] != self.N:
                raise Exception("The timeslot number of the pilot matrix is not same as the given timeslot number.");
            elif self.Xp.shape[-1] != self.M:
                raise Exception("The subcarrier number of the pilot matrix is not same as the given subcarrier number.");
        # opt - the data locs matrix
        if XdLocs is not None:
            self.XdLocs = np.asarray(XdLocs) if self.batch_size is self.BATCH_SIZE_NO else np.tile(XdLocs, (self.batch_size, 1, 1));
            if self.XdLocs.shape[-2] != self.N:
                raise Exception("The timeslot number of the data location matrix is not same as the given timeslot number.");
            elif self.XdLocs.shape[-1] != self.M:
                raise Exception("The subcarrier number of the data location matrix is not same as the given subcarrier number.");
            self.data_len = np.sum(self.XdLocs);
    
    '''
    build the ideal pulse DD channel (callable after modulate)
    @taps_num:  the number of paths
    @his:       the channel gains
    @lis:       the channel delays
    @kis:       the channel dopplers
    '''
    def buildOtfsBiortDDChannel(self, p, his, lis, kis):
        # input check
        if self.pulse_type != self.PUL_BIORT:
            raise Exception("Cannot build the ideal pulse DD channel while not using ideal pulse.");
        hw = self.zeros(self.N, self.M).astype(complex);
        H_DD = self.zeros(self.sig_len, self.sig_len).astype(complex);
        for l in range(self.M):
            for k in range(self.N):
                    for tap_id in range(p):
                        if self.batch_size == self.BATCH_SIZE_NO:
                            hi = his[tap_id];
                            li = lis[tap_id];
                            ki = kis[tap_id];
                        else:
                            hi = np.expand_dims(his[..., tap_id], axis=-1);
                            li = np.expand_dims(lis[..., tap_id], axis=-1);
                            ki = np.expand_dims(kis[..., tap_id], axis=-1);
                        hw_add = 1/self.sig_len*hi*np.exp(-2j*np.pi*li*ki/self.sig_len)* \
                                np.expand_dims(np.sum(np.exp(2j*np.pi*(l-li)*self.seq(self.M)/self.M), axis=-1), axis=-1)* \
                                np.expand_dims(np.sum(np.exp(-2j*np.pi*(k-ki)*self.seq(self.N)/self.N), axis=-1), axis=-1);
                        if self.batch_size == self.BATCH_SIZE_NO:
                            hw[k, l]= hw[k, l] + hw_add;
                        else:
                            hw[..., k, l]= hw[...,k, l] + self.squeeze(hw_add);
                    if self.batch_size == self.BATCH_SIZE_NO:
                        H_DD = H_DD + hw[k, l]*self.kron(self.circshift(self.eye(self.N), k), self.circshift(self.eye(self.M), l));
                    else:
                        H_DD = H_DD + np.expand_dims(hw[..., k, l], axis=(-1,-2))*self.kron(self.circshift(self.eye(self.N), k), self.circshift(self.eye(self.M), l));
        return H_DD;
    
    '''
    build the rectangular pulse DD channel (callable after modulate)
    @taps_num:  the number of paths
    @his:       the channel gains
    @lis:       the channel delays
    @kis:       the channel dopplers
    '''
    def buildOtfsRectaDDChannel(self, p, his, lis, kis):
        # input check
        if self.pulse_type != self.PUL_RECTA:
            raise Exception("Cannot build the rectangular pulse DD channel while not using rectanular pulse.");
        # build H_DD
        H_DD = self.zeros(self.sig_len, self.sig_len);      # intialize the return channel
        dftmat = self.dftmtx(self.N);            # DFT matrix
        idftmat = np.conj(dftmat);                          # IDFT matrix
        piMat = self.eye(self.sig_len);                     # permutation matrix (from the delay) -> pi
        # accumulate all paths
        for tap_id in range(p):
            if self.batch_size == self.BATCH_SIZE_NO:
                hi = his[tap_id];
                li = lis[tap_id];
                ki = kis[tap_id];
            else:
                hi = his[..., tap_id];
                li = lis[..., tap_id];
                ki = np.expand_dims(kis[..., tap_id], axis=-1);
            # delay
            piMati = self.circshift(piMat, li);
            # Doppler            
            deltaMat_diag = np.exp(2j*np.pi*ki/(self.sig_len)*self.buildTimeSequence(li));
            deltaMati = self.diag(deltaMat_diag);
            # Pi, Qi & Ti
            Pi = self.kron(dftmat, self.eye(self.M)) @ piMati;
            Qi = deltaMati @ self.kron(idftmat, self.eye(self.M));
            Ti = Pi @ Qi;
            # add this path
            if self.batch_size == self.BATCH_SIZE_NO:
                H_DD = H_DD + hi*Ti;
            else:
                H_DD = H_DD + hi.reshape(-1, 1, 1)*Ti;
        return H_DD;
    
    '''
    build the time sequence for the given delay
    '''
    def buildTimeSequence(self, li):
        if self.batch_size is self.BATCH_SIZE_NO:
            ts = np.append(np.arange(0, self.sig_len-li), np.arange(-li, 0));
        else:
            if np.all(li == li[0]):
                ts = np.append(np.arange(0, self.sig_len-li[0]), np.arange(-li[0], 0));
                ts = np.tile(ts, (self.batch_size, 1));
            else:
                ts = np.zeros((self.batch_size, self.sig_len), dtype=float);
                for batch_id in range(self.batch_size):
                    ts[batch_id, :] = np.append(np.arange(0, self.sig_len-li[batch_id]), np.arange(-li[batch_id], 0));
        return ts;
    
    '''
    symbol mapping (hard)
    '''
    def symmap(self, syms):
        syms = np.asarray(syms);
        if not self.isvector(syms):
            raise Exception("Symbols must be into a vector form to map.");
        syms_len = syms.shape[-1];
        syms_mat = self.repmat1(np.expand_dims(syms, -1), 1, self.constel_len);
        constel_mat = self.repmat1(self.constel, syms_len, 1);
        syms_dis = abs(syms_mat - constel_mat)**2;
        syms_dis_min_idx = syms_dis.argmin(axis=-1);
        
        return np.take(self.constel, syms_dis_min_idx);
    
    '''
    symbol mapping (hard, no batch)
    '''
    def symmapNoBat(self, syms):
        syms_len = syms.shape[-1];
        syms_mat = np.tile(np.expand_dims(syms, -1), (1, self.constel_len));
        constel_mat = self.repmat1(self.constel, syms_len, 1);
        syms_dis = abs(syms_mat - constel_mat)**2;
        syms_dis_min_idx = syms_dis.argmin(axis=-1);
        return np.take(self.constel, syms_dis_min_idx);