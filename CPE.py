from scipy.stats.distributions import chi2
import numpy as np


class CPE(object):
    # constants
    PIL_VAL = (1+1j)/np.sqrt(2);
    
    # variables
    M = 0; # subcarrier number
    N = 0; # timeslot number
    lmax = 0;
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
    @No: the noise power
    '''
    def __init__(self, otfsconfig, lmax, Es_d, No):
        self.M = otfsconfig.M;
        self.N = otfsconfig.N;
        self.lmax = lmax;
        self.batch_size = otfsconfig.batch_size;
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
        
    '''
    generate pilots
    @Es_p: the pilot energy
    '''
    def genPilots(self, Es_p):
        self.Es_p = Es_p;
        self.pil_val = self.PIL_VAL*np.sqrt(Es_p);
        Xp = np.zeros([self.batch_size, self.N, self.M]).astype(complex);
        for area_id in range(self.area_num):
            Xp[:, self.pk, self.pls[area_id]] = self.pil_val;
        return Xp;
    
    '''
    estimate the path (h, k, l)
    @Y_DD: the received resrouce grid (N, M)
    '''    
    def estPaths(self, Y_DD):
        # we sum all area together
        Y_DD_area_sum = None;
        est_area = np.zeros([self.batch_size, self.N, self.area_len]).astype(complex);
        ca_id_beg = 0;
        ca_id_end = self.area_len - 1;
        # accumulate all areas together
        for area_id in range(self.area_num):
            est_area = est_area + Y_DD[..., :, ca_id_beg:ca_id_end];
            ca_id_beg = ca_id_end + 1;
            ca_id_end = ca_id_end + self.area_len - 1;
        est_area = est_area/self.area_num;
        # find paths
        # estimate the channe
        his = [];
        kis = [];
        lis = [];
        for k_id in range(0, self.N):
            for l_id in range(0, self.area_len):
                pss_ys = np.expand_dims(est_area[k_id, l_id], axis=0) if self.batch_size == self.BATCH_SIZE_NO else est_area[..., k_id, l_id];
                pss_ys_ids_yes = abs(pss_ys)**2 > self.thres;
                pss_ys_ids_not = abs(pss_ys)**2 <= self.thres;
                li = l_id - self.pl;
                ki = k_id - self.pk;
                if self.otfsconfig.isPulIdeal():
                    pss_beta = np.exp(-2j*np.pi*li*ki/self.M/self.N);
                elif self.otfsconfig.isPulRecta():
                    pss_beta = np.exp(2j*np.pi*self.pl*ki/self.M/self.N);
                hi = pss_ys/self.pil_val/pss_beta;
                # at least we find one path
                if np.sum(pss_ys_ids_yes, axis=None) > 0:
                    if self.batch_size == self.BATCH_SIZE_NO:
                        his = np.append(his, hi);
                        lis = np.append(lis, li);
                        kis = np.append(kis, ki);
                    else:
                        hi[pss_ys_ids_not] = 0;
                        hi = hi[..., np.newaxis];
                        li = np.tile(li, (self.batch_size, 1));
                        ki = np.tile(ki, (self.batch_size, 1));
                        if isinstance(his, list):
                            his = hi;
                            lis = li;
                            kis = ki;
                        else:
                            his = np.append(his, hi, axis=-1);
                            lis = np.append(lis, li, axis=-1);
                            kis = np.append(kis, ki, axis=-1);
        return his, lis, kis;