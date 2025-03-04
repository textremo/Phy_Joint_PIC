class OTFSConfig(object):
    ###########################################################################
    # Constants
    # frame type
    FRAME_TYPE_GIVEN = -1;      # user input the frame matrix directly
    FRAME_TYPE_FULL = 0;        # full data
    FRAME_TYPE_CP = 1;          # using cyclic prefix between each two adjacent OTFS subframe (no interference between two adjacent OTFS subframes)
    FRAME_TYPE_ZP = 2;          # using zero padding (no interference between two adjacent OTFS subframes)
    FRAME_TYPES = [FRAME_TYPE_GIVEN, FRAME_TYPE_FULL, FRAME_TYPE_CP, FRAME_TYPE_ZP];
    # pulse type
    PUL_TYPES_IDEAL = 10;       # using ideal pulse to estimate the channel
    PUL_TYPES_RECTA = 20;       # using rectangular waveform to estimate the channel
    PUL_TYPES = [PUL_TYPES_IDEAL, PUL_TYPES_RECTA]; 
    # pilot types
    PIL_TYPE_GIVEN = -1;                        # user input the pilot matrix directly
    PIL_TYPE_NO = 0;                            # no pilots
    PIL_TYPE_EM_MID = 10;                       # embedded pilots - middle
    PIL_TYPE_EM_ZP = 11;                        # embedded pilots - zero padding areas
    PIL_TYPE_SP_MID = 20;                       # superimposed pilots - middle
    PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY = 21;    # superimposed pilots - multiple orthogonal along delay axis
    PIL_TYPES = [PIL_TYPE_GIVEN, PIL_TYPE_NO, PIL_TYPE_EM_MID, PIL_TYPE_EM_ZP, PIL_TYPE_SP_MID, PIL_TYPE_SP_MULTI_ORTH_ALONG_DELAY];
    
    ###########################################################################
    # Variables
    frame_type = FRAME_TYPE_FULL;
    pul_type = PUL_TYPES_RECTA;
    pil_type = PIL_TYPE_NO;
    N = 0;      # timeslot number
    M = 0;      # subcarrier number
    # zero padding length (along delay axis at the end)
    zp_len = 0;
    # pilot length on delay Dopller axes (only available for centralised pilots)
    pk_len = 1;
    pl_len = 1;
    # guard lengths on delay Doppler axes (only available for )
    gl_len_neg = 0;
    gl_len_pos = 0;
    gk_len_neg = 0;
    gk_len_pos = 0;
    # batch_size
    batch_size = None;
    
    '''
    init
    @batch_size: batch_size
    '''
    def __init__(self, *, batch_size = None):
        if batch_size is not None:
            self.batch_size = batch_size;
    
    '''
    set the frame
    @frame_type:    frame type
    @M:             subcarrier number
    @N:             timeslote number
    @zp_len:        zero padding length
    '''
    def setFrame(self, frame_type, M, N, *, zp_len=0):
        if frame_type not in self.FRAME_TYPES:
            raise Exception("`frame_type` must be a type of `OTFSConfig.FRAME_TYPES`.");
        elif frame_type == self.FRAME_TYPE_ZP and (zp_len <= 0 or zp_len >= M):
            raise Exception("In zeor padding OTFS, `zp_len` must be positive and less than subcarrier number.");
        if M <= 0:
            raise Exception("Subcarrier number must be positive.");
        if N <= 0:
            raise Exception("Timeslot number must be positive.");
        self.frame_type = frame_type;
        self.M = M;
        self.N = N;
        self.zp_len = zp_len;
        
    '''
    set the pulse
    @pul_type: pulse type
    '''
    def setPul(self, pul_type):
        if pul_type not in self.PUL_TYPES:
            raise Exception("`pul_type` must be a type of `OTFSConfig.PUL_TYPES`");
        self.pul_type = pul_type;
            
    '''
    set the pilot
    @pil_type: pilote type
    @pk_len: the pilot length along the 
    '''
    def setPil(self, pil_type, *, pk_len=0, pl_len=0):       
        if self.frame_type == self.FRAME_TYPE_FULL and (pil_type == self.PIL_TYPE_EM_MID or pil_type == self.PIL_TYPE_EM_ZP):
            raise Exception("The embedded pilots are not allowed to use while the frame is set to full data.");
        if self.frame_type != self.FRAME_TYPE_CP and (pil_type == self.PIL_TYPE_EM_ZP):
            raise Exception("The zero pading pilots are not allowed to use while the frame is not set to zero padding.");
        # check optional inputs based on pilot types
        self.pil_type = pil_type;
        if pil_type == self.PIL_TYPE_EM_MID or pil_type == self.PIL_TYPE_EM_ZP or pil_type == self.PIL_TYPE_SP_MID:
            if pk_len <= 0 or pk_len >= self.N:
                raise Exception("`pk_len` must be positive and less than the timeslot number.");
            if pl_len <= 0 or pl_len >= self.M:
                raise Exception("`pl_len` must be positive and less than the subcarrier number.");
            self.pk_len = pk_len;
            self.pl_len = pl_len;
            
    '''
    set the guard (only available for embedded pilots)
    '''
    def setGuard(self, gl_len_neg, gl_len_pos, gk_len_neg, gk_len_pos):
        if self.pil_type != self.PIL_TYPE_EM_MID and self.pil_type != self.PIL_TYPE_EM_ZP:
            raise Exception("The guard is only available when using embedded pilots.");
        if self.pk_len <= 0 or self.pk_len >= self.N:
            raise Exception("`pk_len` must be positive and less than the timeslot number.");
        if self.pl_len <= 0 or self.pl_len >= self.M:
            raise Exception("`pl_len` must be positive and less than the subcarrier number.");
        