import numpy as np
from vb_jed import vb_jed_scalar, qpsk_constellation

def run_demo(N=200, snr_db=10, pilot_ratio=0.1, seed=0):
    np.random.seed(seed)
    # constellation
    C = qpsk_constellation()
    S = len(C)

    # generate random bits -> qpsk symbols
    num_data = N
    # pick pilot positions
    P = max(4, int(np.round(pilot_ratio * N)))
    pilot_positions = np.linspace(0, N-1, P, dtype=int)
    # pilot symbols (random)
    pilot_symbols = np.random.choice(C, size=P)

    # true channel (scalar)
    h_true = (np.random.randn() + 1j*np.random.randn())/np.sqrt(2) * 0.8
    # generate data symbols
    data_symbols = np.random.choice(C, size=N)
    # force pilots
    for idx, s in zip(pilot_positions, pilot_symbols):
        data_symbols[idx] = s

    # noise
    # compute signal power
    sig_power = np.mean(np.abs(h_true * data_symbols)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))

    y = h_true * data_symbols + noise

    # run VB-JED
    res = vb_jed_scalar(y, pilot_positions, pilot_symbols,
                        constellation=C, max_iter=200, tol=1e-7,
                        sigma2_init=noise_power, prior_var_h=1.0, estimate_noise=True,
                        damping=0.8, verbose=False)

    # decide symbols by max prob
    qx_probs = res['qx_probs']
    est_symbols = C[np.argmax(qx_probs, axis=1)]
    # compute BER (symbol error)
    ser = np.mean(est_symbols != data_symbols)
    # channel NMSE
    nmse = np.abs(res['m_h'] - h_true)**2 / (np.abs(h_true)**2 + 1e-12)

    print("True h:", h_true)
    print("Estimated h:", res['m_h'], "var:", res['v_h'])
    print("Estimated sigma2:", res['sigma2'])
    print(f"SER: {ser:.4f}, channel NMSE: {nmse:.4e}, iterations: {res['iters']}")

if __name__ == "__main__":
    run_demo(N=400, snr_db=8, pilot_ratio=0.05, seed=42)
