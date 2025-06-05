import numpy as np

'''
generate the full list of H
@kmax:  the kmax in channel estimation (instead of real kmax)
@lmax:  the lmax in channel estimation (instead of real lmax)
'''
def realH2Hfull(kmax, lmax, his, lis, kis, *, batch_size=None):
    his_new = np.zeros((2*kmax+1)*(lmax+1), dtype=complex) if batch_size is None else np.zeros([batch_size, (2*kmax+1)*(lmax+1)], dtype=complex);
    p_len = len(his) if batch_size is None else his.shape[-1];
    for p_id in range(p_len):
        hi = his[p_id] if batch_size is None else his[..., p_id];
        li = lis[p_id] if batch_size is None else lis[..., p_id];
        ki = kis[p_id] if batch_size is None else kis[..., p_id];
        pos = (ki + kmax)*(lmax + 1) + li;
        if pos.ndim < 1:
            his_new[:, pos] = hi;
        else:
            for b_id in range(batch_size):
                his_new[b_id, pos[b_id]] = hi[b_id];

            
    return his_new;