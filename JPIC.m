classdef JPIC < handle
    % constants
    properties(Constant)
        % PUL (pulse)
        PUL_BIORT   = 1;        % bi-orthogonal pulse
        PUL_RECTA   = 2;        % rectangular pulse
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
        SD_DSC_ISE_MMSE    = 1;
        SD_DSC_ISE_MRC     = 2;
        SD_DSC_ISE_LS      = 3;
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
        % Pulse
        pulse_type              = JPIC.PUL_RECTA;
        % MOD
        mod_type                = JPIC.MOD_OTFS;
        % CE
        ce_type                 = JPIC.CE_LS;
        % SD
        sd_bso_mean_cal_init    = JPIC.SD_BSO_MEAN_CAL_INIT_MMSE;
        sd_bso_mean_cal         = JPIC.SD_BSO_MEAN_CAL_MRC;
        sd_bso_var              = JPIC.SD_BSO_VAR_TYPE_ACCUR;
        sd_bso_var_cal          = JPIC.SD_BSO_VAR_CAL_MRC;
        sd_bse                  = JPIC.SD_BSE_OFF;
        sd_dsc_ise              = JPIC.SD_DSC_ISE_MRC;
        sd_dsc_mean_prev_sour   = JPIC.SD_DSC_MEAN_PREV_SOUR_BSE;
        sd_dsc_var_prev_sour    = JPIC.SD_DSC_VAR_PREV_SOUR_BSE;
        sd_out                  = JPIC.SD_OUT_DSC;
        % other settings
        is_early_stop           = false;
        min_var {mustBeNumeric} = eps;      % the default minimal variance is 2.2204e-16
        iter_num                = 10;       % maximal iteration
        iter_diff_min           = eps;      % the minimal difference between 2 adjacent iterations 
        % OTFS configuration
        sig_len = 0;
        data_len = 0;
        M = 0;                              % the subcarrier number
        N = 0;                              % the timeslot number
        Xp = [];                            % pilots values (a matrix)
        XdLocs = [];                       % data locations matrix
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % general methods
    methods
        %{
        constructor
        @constellation:     the constellation, a vector.
        @min_var:           the minimal variance.
        @iter_num:          the maximal iteration.
        @iter_diff_min:     the minimal difference in **DSC** to early stop.
        %}
        function self = JPIC(constel, varargin)
            % take inputs
            if ~isvector(constel)
                error("The constellation must be a vector.");
            else
                % constellation must be a row vector or an 1D vector
                constel = constel(:);
                constel = constel.';
                self.constel = constel;
                self.constel_len = length(constel);
                self.es = sum(abs(constel).^2)/self.constel_len;   % constellation average power
            end
            %TODO: add optional inputs
        end
        
        %{
        settings - pulse type
        %}
        function setPul2Biort(self)
            self.pulse_type = self.PUL_BIORT;
        end
        function setPul2Recta(self)
            self.pulse_type = self.PUL_RECTA;
        end

        %{
        settings - MOD
        %}
        function setMod2Ofdm(self)
            self.mod_type = self.MOD_MIMO;
        end
        %{
        settings - MOD - OTFS
        @M:             subcarrier number
        @N:             timeslot number
        @Xp(opt):       the pilot value matrix ([batch_size], N, M)
        @XdLocs(opt):   the data locs matrix ([batch_size], N, M)
        %}
        function setMod2Otfs(self, M, N)
            self.mod_type = self.MOD_OTFS;
            self.setOTFS(M, N);
        end
        function setMod2OtfsEM(self, M, N, varargin)
            self.mod_type = self.MOD_OTFS_EM;
            self.setOTFS(M, N, varargin{:});
        end
        function setMod2OtfsSP(self, M, N, varargin)
            self.mod_type = self.MOD_OTFS_SP;
            self.setOTFS(M, N, varargin{:});
        end

        %{
        settings - CE
        %}
        function setCE2LS(self)
            self.ce_type = self.CE_LS;
        end
        
        %{
        settings - SD
        %}
        % settings - SD - BSO - mean - cal (init)
        function setSdBsoMealCalInit2MMSE(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MMSE;
        end
        function setSdBsoMealCalInit2MRC(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_MRC;
        end
        function setSdBsoMealCalInit2LS(self)
            self.sd_bso_mean_cal_init = self.SD_BSO_MEAN_CAL_INIT_LS;
        end
        % settings - SD - BSO - mean - cal
        function setSdBsoMeanCal2MRC(self)
            self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_MRC;    
        end
        function setSdBsoMeanCal2LS(self)
            self.sd_bso_mean_cal = self.SD_BSO_MEAN_CAL_LS;
        end
        % settings - SD - BSO - var
        function setSdBsoVar2Appro(self)
            self.sd_bso_var = self.SD_BSO_VAR_TYPE_APPRO;
        end
        function setSdBsoVar2Accur(self)
            self.sd_bso_var = self.SD_BSO_VAR_TYPE_ACCUR;
        end
        % settings - SD - BSO - var - cal
        function setSdBsoVarCal2MMSE(self)
            self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MMSE;
        end
        function setSdBsoVarCal2MRC(self)
            self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_MRC;
        end
        function setSdBsoVarCal2LS(self)
            self.sd_bso_var_cal = self.SD_BSO_VAR_CAL_LS;
        end
        % settings - SD - BSE
        function setSdBseOn(self)
            self.sd_bse = self.SD_BSE_ON;
        end
        function setSdBseOff(self)
            self.sd_bse = self.SD_BSE_OFF;
        end
        % settings - SD - DSC - instantaneous square error
        function setSdDscIse2MMSE(self)
            self.sd_dsc_ise = self.SD_DSC_ISE_MMSE;
        end
        function setSdDscIse2MRC(self)
            self.sd_dsc_ise = self.SD_DSC_ISE_MRC;
        end
        function setSdDscIse2LS(self)
            self.sd_dsc_ise = self.SD_DSC_ISE_LS;
        end
        % settings - SD - DSC - mean previous source
        function setSdDscMeanPrevSour2BSE(self)
            self.sd_dsc_mean_prev_sour = self.SD_DSC_MEAN_PREV_SOUR_BSE;
        end
        function setSdDscMeanPrevSour2DSC(self)
            self.sd_dsc_mean_prev_sour = self.SD_DSC_MEAN_PREV_SOUR_DSC;
        end
        % settings - SD - DSC - variance previous source
        function setSdDscVarPrevSour2BSE(self)
            self.sd_dsc_var_prev_sour = self.SD_DSC_VAR_PREV_SOUR_BSE;
        end
        function setSdDscVarPrevSour2DSC(self)
            self.sd_dsc_var_prev_sour = self.SD_DSC_VAR_PREV_SOUR_DSC;
        end
        % settings - SD - OUT
        function setSdOut2BSE(self)
            if self.sd_bse == self.SD_BSE_OFF
                error("Cannot use BSE output because BSE module is off.");
            end
            self.sd_out = self.SD_OUT_BSE;
        end
        function setSdOut2DSC(self)
            self.sd_out = self.SD_OUT_DSC;
        end

        %{
        detect
        @Y_DD:          the received signal in the delay Doppler domain [(batch_size), doppler, delay]
        @lmax:          the maximal delay index
        @kmax:          the maximal Doppler index
        @No:            the noise (linear) power
        @sym_map(opt):  false by default. If true, the output will be mapped to the constellation
        %}
        function [x, H] = detect(self, Y_DD, lmax, kmax, No, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"sym_map", false, @(x) isscalar(x)&islogical(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            sym_map = inPar.Results.sym_map;

            % input check
            [Y_DD_N, Y_DD_M] = size(Y_DD);
            if Y_DD_N ~= self.N || Y_DD_M ~= self.M
                error("The received matrix is not in the shape of (%d, %d).", self.N, self.M);
            end
            if ~isscalar(No)
                error("The noise power is not a scalar.");
            end

            % constant values
            y = Y_DD.';
            y = y(:);
            xp = self.Xp.';
            xp = xp(:);
            xdlocs = self.XdLocs.';
            xdlocs = xdlocs(:);
            
            % iterative detection
            Xe = zeros(self.N, self.M);
            ise_dsc_prev = zeros(self.sig_len, 1);
            x_bse = zeros(self.sig_len, 1);
            v_bse = zeros(self.sig_len, 1);
            x_dsc = zeros(self.sig_len, 1);
            v_dsc = zeros(self.sig_len, 1);
            x_det = zeros(self.sig_len, 1);
            for iter_id = 1:self.iter_num
                % CE
                X = Xe + self.Xp;
                Phi = self.buildPhi(X, lmax, kmax);
                h = inv(Phi'*Phi)*Phi'*y;
                % build the channel
                H = self.buildHdd(h, lmax, kmax);
                [~, x_num] = size(H);
                Ht = H';
                Hty = Ht*y;
                HtH = Ht*H;
                HtH_off = ((eye(x_num)+1) - eye(x_num).*2).*HtH;
                HtH_off_sqr = HtH_off.^2;
                % SD
                % SD - BSO
                % SD - BSO - mean
                if iter_id == 1
                    % SD - BSO - mean - 1st iter
                    switch self.sd_bso_mean_cal_init
                        case self.SD_BSO_MEAN_CAL_INIT_MMSE
                            bso_zigma_1 = inv(HtH + No/self.es*eye(x_num));
                        case self.SD_BSO_MEAN_CAL_INIT_MRC
                            bso_zigma_1 = diag(1./vecnorm(H).^2);
                        case self.SD_BSO_MEAN_CAL_INIT_LS
                            bso_zigma_1 = inv(HtH);
                        otherwise
                            bso_zigma_1 = eye(x_num);
                    end
                    x_bso = bso_zigma_1*(Hty - HtH_off*x_dsc);
                else
                    % SD - BSO - mean - other iteration
                    switch self.sd_bso_mean_cal
                        case self.SD_BSO_MEAN_CAL_MRC
                            bso_zigma_n = diag(1./vecnorm(H).^2);
                        case self.SD_BSO_MEAN_CAL_LS
                            bso_zigma_n = inv(HtH);
                    end
                    x_bso = bso_zigma_n*(Hty - HtH_off*x_dsc);
                end
                % SD - BSO - variance
                switch self.sd_bso_var_cal
                    case self.SD_BSO_VAR_CAL_MMSE
                        bso_var_mat = diag(inv(HtH + No/self.es*eye(x_num)));
                    case self.SD_BSO_VAR_CAL_MRC
                        bso_var_mat = 1./vecnorm(H).^2.';
                    case self.SD_BSO_VAR_CAL_LS
                        bso_var_mat = diag(inv(HtH));
                end
                bso_var_mat_sqr = bso_var_mat.^2;
                if self.sd_bso_var == self.SD_BSO_VAR_TYPE_APPRO
                    v_bso = No.*bso_var_mat;
                end
                if self.sd_bso_var == self.SD_BSO_VAR_TYPE_ACCUR
                    v_bso = No.*bso_var_mat + HtH_off_sqr*v_dsc.*bso_var_mat_sqr;
                end
                v_bso = max(v_bso, self.min_var);

                % SD - BSE
                bse_x_bso_mat = repmat(x_bso(xdlocs), 1, self.constel_len);
                bse_constel_mat = repmat(self.constel, self.data_len, 1) + repmat(xp(xdlocs), 1, self.constel_len);
                % BSE - Estimate P(x|y) using Gaussian distribution
                pxyPdfExpPower = -1./(2*v_bso(xdlocs)).*abs(bse_x_bso_mat - bse_constel_mat).^2;
                pxypdfExpNormPower = pxyPdfExpPower - max(pxyPdfExpPower, [], 2);   % make every row the max power is 0
                pxyPdf = exp(pxypdfExpNormPower);
                % BSE - Calculate the coefficient of every possible x to make the sum of all
                pxyPdfCoeff = 1./sum(pxyPdf, 2);
                pxyPdfCoeff = repmat(pxyPdfCoeff, 1, self.constel_len);
                % BSE - PDF normalisation
                pxyPdfNorm = pxyPdfCoeff.*pxyPdf;
                % BSE - calculate the mean and variance
                x_bse(xdlocs) = sum(pxyPdfNorm.*self.constel, 2);
                x_bse_mat = repmat(x_bse(xdlocs), 1, self.constel_len);
                v_bse(xdlocs) = sum(abs(x_bse_mat - self.constel).^2.*pxyPdfNorm, 2);
                v_bse(xdlocs) = max(v_bse(xdlocs), self.min_var);
                % BSE - to original size

                % SD - DSC
                switch self.sd_dsc_ise
                    case self.SD_DSC_ISE_MMSE
                        dsc_w = inv(HtH + No/self.es*eye(x_num));
                    case self.SD_DSC_ISE_MRC
                        dsc_w = diag(1./vecnorm(H).^2);
                    case self.SD_DSC_ISE_LS
                        dsc_w = inv(HtH);
                end
                ise_dsc = (dsc_w*(Hty - HtH*(x_bse + xp))).^2;
                ies_dsc_sum = ise_dsc + ise_dsc_prev;
                ies_dsc_sum = max(ies_dsc_sum, self.min_var);
                % DSC - rho (if we use this rho, we will have a little difference)
                rho_dsc = ise_dsc_prev./ies_dsc_sum;
                % DSC - mean
                if iter_id == 1
                    x_dsc = x_bse;
                else
                    if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_BSE
                        %x_dsc = ise_dsc./ies_dsc_sum.*x_bse_prev + ise_dsc_prev./ies_dsc_sum.*x_bse;
                        x_dsc = (1 - rho_dsc).*x_bse_prev + rho_dsc.*x_bse;
                    end
                    if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_DSC
                        x_dsc = (1 - rho_dsc).*x_dsc + rho_dsc.*x_bse;
                    end
                end
                % DSC - variance
                if iter_id == 1
                    v_dsc = v_bse;
                else
                    if self.sd_dsc_var_prev_sour == self.SD_DSC_VAR_PREV_SOUR_BSE
                        %v_dsc = ise_dsc./ies_dsc_sum.*v_bse_prev + ise_dsc_prev./ies_dsc_sum.*v_bse;
                        v_dsc = (1 - rho_dsc).*v_bse_prev + rho_dsc.*v_bse;
                    end
                    if self.sd_dsc_var_prev_sour == self.SD_DSC_VAR_PREV_SOUR_DSC
                        v_dsc = (1 - rho_dsc).*v_dsc + rho_dsc.*v_bse;
                    end
                end

                % early stop
                if iter_id > 1 && sum(abs(v_dsc - v_dsc_prev).^2) <= self.iter_diff_min
                    break;
                end

                % update statistics
                % update statistics - BSE
                if self.sd_dsc_mean_prev_sour == self.SD_DSC_MEAN_PREV_SOUR_BSE
                    x_bse_prev = x_bse;
                end
                if self.sd_dsc_var_prev_sour == self.SD_DSC_VAR_PREV_SOUR_BSE
                    v_bse_prev = v_bse;
                end
                % update statistics - DSC
                v_dsc_prev = v_dsc;
                % update statistics - DSC - instantaneous square error
                ise_dsc_prev = ise_dsc;

                % soft symbol estimation
                % take the detection value
                if self.sd_out == self.SD_OUT_BSE
                    x = x_bse;
                end
                if self.sd_out == self.SD_OUT_DSC
                    x = x_dsc;
                end
                x(xdlocs) = self.symmap(x(xdlocs));
                Xe = reshape(x, self.M, self.N).';
            end
            % only keep data part
            x = x(xdlocs);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Auxiliary Methods
    methods
        %{
        build Phi - the channel estimation matrix
        @X:     the Tx matrix in DD domain ([batch_size], doppler, delay)
        @lmax:  the maximal delay
        @kmax:  the maximal Doppler
        %}
        function Phi = buildPhi(self, X, lmax, kmax)
            pmax = (lmax+1)*(2*kmax+1);                 % the number of all possible paths
            lis = kron(0:lmax, ones(1, 2*kmax + 1));    % the delays on all possible paths
            kis = repmat(-kmax:kmax, 1, lmax+1);        % the dopplers on all possible paths
            Phi = zeros(self.sig_len, pmax);            % the return matrix
            for yk = 1:self.N
                for yl = 1:self.M
                    Phi_ri = (yk - 1)*self.M + yl;      % row id in Phi
                    for p_id = 1:pmax
                        % path delay and doppler
                        li = lis(p_id);
                        ki = kis(p_id);
                        % x(k, l)_
                        xl = yl - li;
                        if yl-1 < li
                            xl = xl + self.M;
                        end
                        xk = mod(yk - 1 - ki, self.N) + 1;
                        % exponential part (pss_beta)
                        if self.pulse_type == self.PUL_BIORT
                            pss_beta = exp(-2j*pi*li*ki/self.M/self.N);
                        elseif self.pulse_type == self.PUL_RECTA
                            pss_beta = exp(2j*pi*(yl - li - 1)*ki/self.M/self.N); % here, you must use `yl-li-1` instead of `xl-1` or there will be an error
                            if yl-1 < li
                                pss_beta = pss_beta*exp(-2j*pi*(xk-1)/self.N);
                            end
                        end
                        % assign value
                        Phi(Phi_ri, p_id) = X(xk, xl)*pss_beta;
                    end
                end
            end  
        end

        %{
        build Hdd
        @his: the path gains
        @lmax:  the maximal delay
        @kmax:  the maximal Doppler
        @thres(opt): the threshold of a path (default 0)
        %}
        function Hdd = buildHdd(self, his, lmax, kmax, varargin)
            % register optional inputs 
            inPar = inputParser;
            addParameter(inPar,"thres", 0, @(x) isscalar(x)&isnumeric(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            thres = inPar.Results.thres;
            % other inputs
            pmax = (lmax+1)*(2*kmax+1);                 % the number of all possible paths
            lis = kron(0:lmax, ones(1, 2*kmax + 1));    % the delays on all possible paths
            kis = repmat(-kmax:kmax, 1, lmax+1);        % the dopplers on all possible paths
            % filter the path gains
            his(abs(his) < abs(thres)) = 0;
            % build the channel in DD domain
            switch self.pulse_type
                case self.PUL_BIORT
                    Hdd = self.buildOtfsBiortDDChannel(pmax, his, lis, kis);
                case self.PUL_RECTA
                    Hdd = self.buildOtfsRectaDDChannel(pmax, his, lis, kis);
                otherwise
                    error("The pulse type is not recognised.");
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % private methods
    methods(Access=private)
        %{
        set OTFS
        @M:             subcarrier number
        @N:             timeslot number
        @Xp(opt):       the pilot value matrix ([batch_size], N, M)
        @XdLocs(opt):   the data locs matrix ([batch_size], N, M)
        %}
        function setOTFS(self, M, N, varargin)
            % OTFS size
            if M < 1 || floor(M) ~= M
                error("The subcarrier number cannot be less than 1 or a fractional number.");
            else
                self.M = M;
            end
            if N < 1 || floor(N) ~= N
                error("The timeslot number cannot be less than 1 or a fractional number.");
            else
                self.N = N;
            end
            self.sig_len = N*M;
            self.data_len = N*M;
            % opt
            inPar = inputParser;
            addParameter(inPar,"Xp", [], @(x) isempty(x)|ismatrix(x)&isnumeric(x));
            addParameter(inPar,"XdLocs", [], @(x) isempty(x)|ismatrix(x)&islogical(x));
            inPar.KeepUnmatched = true;
            inPar.CaseSensitive = false;
            parse(inPar, varargin{:});
            % opt - the pilot value matrix
            self.Xp = inPar.Results.Xp;
            if ~isempty(self.Xp)
                [XpN, XpM] = size(self.Xp);
                if XpN ~= self.N
                    error("The timeslot number of the pilot matrix is not same as the given timeslot number.");
                elseif XpM ~= self.M
                    error("The subcarrier number of the pilot matrix is not same as the given subcarrier number.");
                end
            end
            % opt - the data locs matrix
            self.XdLocs = inPar.Results.XdLocs;
            if ~isempty(self.XdLocs)
                [XdLocsN, XdLocsM] = size(self.XdLocs);
                if XdLocsN ~= self.N
                    error("The timeslot number of the pilot matrix is not same as the given timeslot number.");
                elseif XdLocsM ~= self.M
                    error("The subcarrier number of the pilot matrix is not same as the given subcarrier number.");
                end
                self.data_len = sum(self.XdLocs, "all");
            end
        end

        

        %{
        build the ideal pulse DD channel (callable after modulate)
        @taps_num:  the number of paths
        @his:       the channel gains
        @lis:       the channel delays
        @kis:       the channel dopplers
        %}
        function H_DD = buildOtfsBiortDDChannel(self, p, his, lis, kis)
            % input check
            if self.pulse_type ~= self.PUL_BIORT
                error("Cannot build the bi-orthgonal DD channel while not using bi-orthgonal pulse.");
            end
            hw = zeros(self.N, self.M);
            H_DD = zeros(self.sig_len, self.sig_len);
            for l = 1:self.M
                for k = 1:self.N
                    for tap_id = 1:p
                        hi = his(tap_id);
                        li = lis(tap_id);
                        ki = kis(tap_id);
                        hw_add = 1/self.sig_len*hi*exp(-2j*pi*li*ki/self.sig_len)* ...
                                sum(exp(2j*pi*(l-li)*(0:self.M-1)/self.M))* ...
                                sum(exp(-2j*pi*(k-ki)*(0:self.N-1)/self.N));
                        hw(k, l)= hw(k, l) + hw_add;
                    end
                    H_DD = H_DD + hw(k, l)*kron(circshift(eye(self.N), k), circshift(eye(self.M), l));
                end
            end
        end

        %{
        build the rectangular pulse DD channel (callable after modulate)
        @taps_num:  the number of paths
        @his:       the channel gains
        @lis:       the channel delays
        @kis:       the channel dopplers
        %}
        function H_DD = buildOtfsRectaDDChannel(self, p, his, lis, kis)
            % input check
            if self.pulse_type ~= self.PUL_RECTA
                error("Cannot build the rectangular pulse DD channel while not using rectanular pulse.");
            end
            % build H_DD
            H_DD = zeros(self.sig_len, self.sig_len);                       % intialize the return channel
            dftmat = dftmtx(self.N)/sqrt(self.N);                           % DFT matrix
            idftmat = conj(dftmat);                                         % IDFT matrix 
            piMat = eye(self.sig_len);                                      % permutation matrix (from the delay) -> pi
            % accumulate all paths
            for tap_id = 1:p
                hi = his(tap_id);
                li = lis(tap_id);
                ki = kis(tap_id);
                % delay
                piMati = circshift(piMat, li); 
                % Doppler
                deltaMat_diag = exp(2j*pi*ki/(self.sig_len)*self.buildTimeSequence(self.sig_len, li));
                deltaMati = diag(deltaMat_diag);
                % Pi, Qi, & Ti
                Pi = kron(dftmat, eye(self.M))*piMati; 
                Qi = deltaMati*kron(idftmat, eye(self.M));
                Ti = Pi*Qi;
                H_DD = H_DD + hi*Ti;
            end
        end

        %{
        build the time sequence for the given delay
        %}
        function ts = buildTimeSequence(~, sig_len, li)
            ts = [0:sig_len-1-li, -li:-1];
        end

        %{
        symbol mapping (soft)
        %}
        function syms_mapped = symmap(self, syms)
            if ~isvector(syms)
                error("Symbols must be into a vector form to map.");
            end
            % the input must be a column vector
            is_syms_col = iscolumn(syms);
            syms = syms(:);
            syms_dis = abs(syms - self.constel).^2;
            [~,syms_dis_min_idx] =  min(syms_dis,[],2);
            syms_mapped = self.constel(syms_dis_min_idx);
            if is_syms_col
                syms_mapped = syms_mapped(:);
            end
        end
    end

end