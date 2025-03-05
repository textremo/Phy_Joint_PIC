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
        area_len = lmax + 1;
        self.area_num = np.floor(M / area_len).astype(int);
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
        pil_val = self.PIL_VAL*np.sqrt(Es_p);
        Xp = np.zeros([self.batch_size, self.N, self.M]).astype(complex);
        for area_id in range(self.area_num):
            Xp[:, self.pk, self.pls[area_id]] = pil_val;
        return Xp;
    
    '''
    estimate the path (h, k, l)
    @Y_DD: the received resrouce grid (N, M)
    @Es_d: the data energy
    '''    
    def estPaths(self, Y_DD, Es_d):
        # we sum all area together
        Y_DD_area_sum = None;
        
        
        
        pass