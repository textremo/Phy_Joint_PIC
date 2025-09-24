classdef Utils < handle
    methods(Static)
        %{
        generate the full list of H
        @lmax:  the lmax in channel estimation (instead of real lmax)
        @kmax:  the kmax in channel estimation (instead of real kmax)
        %}
        function his_new = realH2Hfull(kmax, lmax, his, lis, kis)
            his_new = zeros(1, (2*kmax+1)*(lmax+1));
            p_len = length(his);
            for p_id = 1:p_len
                hi = his(p_id);
                li = lis(p_id);
                ki = kis(p_id);
                pos = li*(2*kmax+1) + kmax + ki + 1;
                his_new(pos) = hi;
            end
        end
    end
end