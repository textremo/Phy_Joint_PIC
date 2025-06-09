from scipy.stats.distributions import chi2
import numpy as np


class CPE(object):
    # constants
    PIL_VAL = (1+1j)/np.sqrt(2);
    BATCH_SIZE_NO = None;
    
    # variables
    otfsconfig = None;
    M = 0; # subcarrier number
    N = 0; # timeslot number
    lmax = 0;
    kmax = 0
    area_num = 0;
    pl = 0;
    pk = 0;
    pls = [];
    Es_p = 0;
    batch_size = 0;
    # variables - pilots
    pil_val = None;
    area_len = 0;
    rho = 0;        # the probability that the signal is 
    thres = 0;      # the threshold that this point is not noise (in power)
    
    '''
    @otfsconfig: OTFS configuration
    @lmax: the maximal delay index
    @kmax: the maximal Doppler index
    @No: the noise power
    '''
    def __init__(self, otfsconfig, lmax, kmax, Es_d, No, *, B=None):
        self.otfsconfig = otfsconfig;
        self.M = otfsconfig.M;
        self.N = otfsconfig.N;
        self.lmax = lmax;
        self.kmax = kmax
        self.area_len = lmax + 1;
        self.area_num = np.floor(otfsconfig.M / self.area_len).astype(int);
        if self.area_num <= 0:
            raise Exception("There is no space for pilots.");
        # calculate the Doppler positions
        self.pk = np.floor((self.N - 1)/2).astype(int);
        # calculate the delay positions
        pl = 0;
        for area_id in range(self.area_num):
            self.pls.append(pl);
            pl = pl + lmax + 1;
        # calculate the threshold
        self.rho = chi2.ppf(0.9999, 2*self.area_num)/(2*self.area_num);
        self.thres = self.rho*(Es_d + No);
        self.batch_size = B
        
    '''
    generate pilots
    @Es_p: the pilot energy
    '''
    def genPilots(self, Es_p):
        self.Es_p = Es_p;
        self.pil_val = self.PIL_VAL*np.sqrt(Es_p);
        Xp = np.zeros([self.N, self.M], dtype=complex) if self.batch_size is self.BATCH_SIZE_NO else np.zeros([self.batch_size, self.N, self.M], dtype=complex)
        for area_id in range(self.area_num):
            Xp[..., self.pk, self.pls[area_id]] = self.pil_val;
        return Xp;
    
    '''
    estimate the path (h, k, l)
    @Y_DD: the received resrouce grid (N, M)
    @isAll(opt):    estimate all paths  
    '''    
    def estPaths(self, Y_DD, *, isAll=False):
        # we sum all area together
        est_area = np.zeros([self.N, self.area_len], dtype=complex) if self.batch_size is self.BATCH_SIZE_NO else np.zeros([self.batch_size, self.N, self.area_len], dtype=complex);
        ca_id_beg = 0;
        ca_id_end = self.area_len;
        # accumulate all areas together
        for area_id in range(self.area_num):
            # build the phase matrix
            est_area = est_area + Y_DD[..., :, ca_id_beg:ca_id_end]*self.getPhaseConjMat(area_id);
            ca_id_beg = ca_id_end;
            ca_id_end = ca_id_end + self.area_len;
        est_area = est_area/self.area_num;
        # locate the searching space for ki
        ki_beg = self.pk - self.kmax if self.pk - self.kmax > 0 else 0 
        ki_end = self.pk + self.kmax + 1 if self.pk + self.kmax < self.N else self.N
        # find paths
        his = [];
        his_mask = [];
        kis = [];
        lis = [];
        for l_id in range(0, self.area_len):
            for k_id in range(ki_beg, ki_end):
                pss_ys = np.expand_dims(est_area[k_id, l_id], axis=0) if self.batch_size == self.BATCH_SIZE_NO else est_area[..., k_id, l_id];
                pss_ys_ids_yes = abs(pss_ys)**2 > self.thres;
                pss_ys_ids_not = abs(pss_ys)**2 <= self.thres;
                li = l_id - self.pl;
                ki = k_id - self.pk;
                hi = pss_ys/self.pil_val;
                # zero values under the threshold
                hi[pss_ys_ids_not] = 0; 
                hi_mask = pss_ys_ids_yes;
                # at least we find one path
                if isAll or np.sum(pss_ys_ids_yes, axis=None) > 0:
                    if self.batch_size != self.BATCH_SIZE_NO:
                        hi = hi[..., np.newaxis]
                        hi_mask = hi_mask[..., np.newaxis]
                        li = np.tile(li, (self.batch_size, 1));
                        ki = np.tile(ki, (self.batch_size, 1));
                    his.append(hi)     
                    his_mask.append(hi_mask)
                    lis.append(li)
                    kis.append(ki)
        his = np.concatenate(his, -1)
                    
        # return
        if isAll:
            his_mask = np.concatenate(his_mask, -1)
            return his, his_mask
        else:
            lis = np.asarray(lis) if self.batch_size == self.BATCH_SIZE_NO else np.concatenate(lis, -1)
            kis = np.asarray(kis) if self.batch_size == self.BATCH_SIZE_NO else np.concatenate(kis, -1)
            return his, lis, kis;
        
    
    '''
    build the phase Conj matrix
    @area_id: the area index
    '''
    def getPhaseConjMat(self, area_id):
        if area_id < 0 or area_id >= self.area_num:
            raise Exception("Area Id is illegal.");
        phaseConjMat = np.zeros((self.N, self.area_len), dtype=complex);
        for ki in range(0, self.N):
            for li in range(0, self.area_len):
                if self.otfsconfig.isPulIdeal():
                    phaseConjMat[ki, li] = np.exp(2j*np.pi*li*(ki-self.pk)/self.M/self.N);
                elif self.otfsconfig.isPulRecta():
                    phaseConjMat[ki, li] = np.exp(-2j*np.pi*self.pls[area_id]*(ki-self.pk)/self.M/self.N);
                else:
                    raise Exception("The pulse type is unkownn.");
        if self.batch_size is not self.BATCH_SIZE_NO:
            phaseConjMat = np.tile(phaseConjMat, (self.batch_size, 1, 1));
        
        return phaseConjMat;
        
            
    
    '''
    check whether the channel path number are equal
    '''
    def chkChPathNum(self, his):
        pass