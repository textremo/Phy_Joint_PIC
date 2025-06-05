import numpy as np

def zero_forcing_with_constraints(H, y, zero_indices):
    """
    H: M x N channel matrix
    y: M x 1 received vector
    zero_indices: list or array of indices in x that are known zero
    
    Returns:
    x_hat: N x 1 estimated x vector with constraints
    """
    M, N = H.shape
    zero_indices = np.array(zero_indices)
    
    # Construct constraint matrix C
    # Each row selects one zero index in x
    C = np.zeros((N, N))
    for idx in zero_indices:
        C[idx, idx] = 1
    
    # Build big matrix and RHS for KKT system
    HtH = H.conj().T @ H
    Hty = H.conj().T @ y.reshape(-1,1)
    
    KKT_mat = np.block([
        [HtH, C.conj().T],
        [C, np.zeros((N, N))]
    ])
    
    KKT_rhs = np.vstack([Hty, np.zeros((N,1))])
    
    # Solve KKT system
    sol = np.linalg.solve(KKT_mat, KKT_rhs)
    
    x_hat = sol[:N].flatten()
    return x_hat

# Example usage:
n = np.random.randn(6) + 1j*np.random.randn(6)
H = np.random.randn(6, 4) + 1j*np.random.randn(6, 4)
x = np.random.randn(4) + 1j*np.random.randn(4)
zero_idx = [1, 3]  # x[1] and x[3] known zero
for zero_id in zero_idx:
    x[zero_id] = 0;
y = H @ x + n;

x_est = zero_forcing_with_constraints(H, y, zero_idx)
x_est[abs(x_est) <= 1e-13] = 0;

