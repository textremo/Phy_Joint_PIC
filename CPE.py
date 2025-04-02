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
    pk = 0;
    pls = [];
    Es_p = 0;
    batch_size = 0;
    # variables - pilots
    pil_val = None;
    area_len = 0;
    rho0 = 0;       # the probability that the signal is 
    rho1 = 0;
    
    
    '''
    @M: subcarrier number
    @N: timeslote number
    @lmax: the maximal delay index
    '''
    def __init__(self, M, N, lmax, *, batch_size):
        self.M = M;
        self.N = N;
        self.lmax = lmax;
        self.batch_size = batch_size;
        self.area_len = lmax + 1;
        self.area_num = np.floor(M / self.area_len).astype(int);
        if self.area_num <= 0:
            raise Exception("There is no space for pilots.");
        # calculate the Doppler positions
        self.pk = np.floor((self.N - 1)/2).astype(int);
        # calculate the delay positions
        pl = 0;
        for area_id in range(self.area_num):
            self.pls.append(pl);
            pl = pl + lmax + 1;
            
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
    @Es_d: the data energy
    '''    
    def estPaths(self, Y_DD, Es_d):
        # we sum all area together
        Y_DD_area_sum = None;
        est_area = np.zeros([self.batch_size, self.N, (self.lmax+1)]).astype(complex);
        ca_id_beg = 0;
        ca_id_end = self.area_len - 1;
        # accumulate all areas together
        for area_id in range(self.area_num):
            est_area = est_area + Y_DD[..., :, ca_id_beg:ca_id_end];
            ca_id_beg = ca_id_end + 1;
            ca_id_end = ca_id_end + self.area_len - 1;
        # find paths
        for k_id in range(0, self.N):
            for l_id in range(0, self.M):
                if
        
    