import numpy as np

def qpsk_constellation():
    return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

def soft_demod_update(y, m_h, v_h, sigma2, constellation, prior=None):
    """
    更新单个符号的 q(x) (离散 distribution over constellation).
    y: received scalar
    m_h: mean of q(h) (complex)
    v_h: variance of q(h) (real >=0)
    sigma2: noise variance
    constellation: array of complex constellation points
    prior: prior probabilities over constellation (same length), or None->uniform
    返回：probabilities (len const)
    """
    if prior is None:
        prior = np.ones(len(constellation)) / len(constellation)
    # compute log-likelihood up to constant:
    # E_{q(h)}[ |y - h s|^2 ] = |y|^2 - 2 Re( conj(y) m_h s ) + |s|^2 (|m_h|^2 + v_h)
    y_abs2 = np.abs(y)**2
    mh_abs2 = np.abs(m_h)**2
    s = constellation
    # vectorized:
    term = - (1.0 / sigma2) * (
        y_abs2 - 2.0 * np.real(np.conj(y) * m_h * s) + (np.abs(s)**2) * (mh_abs2 + v_h)
    )
    logp = np.log(prior + 1e-20) + term
    # avoid overflow
    logp = logp - np.max(logp)
    p = np.exp(logp)
    p = p / np.sum(p)
    return p

def update_qh(y_all, qx_means, qx_second_moments, sigma2, prior_var_h):
    """
    更新 q(h) —— 复高斯后验 (mean m_h, variance v_h)
    模型 y_n = h * x_n + noise
    对于全部符号集合:
      v_h = 1 / (1/prior_var_h + (1/sigma2) * sum E[|x_n|^2])
      m_h = v_h * (1/sigma2) * sum conj(E[x_n]) * y_n
    注意复数共轭等
    """
    denom = 1.0 / prior_var_h + (1.0 / sigma2) * np.sum(qx_second_moments)
    v_h = 1.0 / denom
    # compute numerator: (1/sigma2) * sum conj(x_mean) * y
    numerator = (1.0 / sigma2) * np.sum(np.conj(qx_means) * y_all)
    m_h = v_h * numerator
    return m_h, v_h

def estimate_sigma2(y_all, m_h, v_h, qx_means, qx_second_moments):
    """
    简单的 noise variance update: sigma2 = (1/N) * sum E[|y_n - h x_n|^2]
    where expectation over q(h) and q(x_n) used.
    E[|y - h x|^2] = |y|^2 - 2 Re(conj(y) E[h] E[x]) + E[|h|^2] E[|x|^2]
    E[|h|^2] = |m_h|^2 + v_h
    """
    N = len(y_all)
    yh = np.abs(y_all)**2 - 2.0 * np.real(np.conj(y_all) * m_h * qx_means) + ( (np.abs(m_h)**2 + v_h) * qx_second_moments )
    sigma2 = np.sum(yh) / N
    # keep positive
    return np.maximum(sigma2, 1e-12)

def vb_jed_scalar(y, pilot_positions, pilot_symbols, constellation=None,
                  max_iter=50, tol=1e-6, sigma2_init=None, prior_var_h=1.0,
                  estimate_noise=True, damping=0.7, verbose=False):
    """
    一个最小可运行的 VB-JED（标量信道）实现。
    Args:
      y: received vector (length N) complex
      pilot_positions: indices (list or array) where pilots are present
      pilot_symbols: values of pilots at pilot_positions (same length)
      constellation: array of complex constellation; if None, use QPSK
      max_iter: max VB iterations
      tol: ELBO/parameter tol (这里用 m_h 变化做简单判定)
      sigma2_init: initial noise variance (if None, use sample var)
      prior_var_h: prior variance of h (complex Gaussian zero-mean)
      estimate_noise: whether update sigma2 each iter
      damping: damping factor for updates (0<d<=1) -> new = damping * new + (1-damping) * old
    Returns:
      dict with keys: 'm_h','v_h','qx_probs' (N x |const|), 'qx_means','sigma2', 'iters'
    """
    y = np.asarray(y)
    N = len(y)
    if constellation is None:
        constellation = qpsk_constellation()
    S = len(constellation)

    # initialization
    if sigma2_init is None:
        sigma2 = np.var(y) * 0.1 + 1e-6
    else:
        sigma2 = sigma2_init

    # q(x): for pilot positions set as delta on pilot symbol
    qx_probs = np.ones((N, S)) / S
    # set pilots
    for idx, sym in zip(pilot_positions, pilot_symbols):
        # find nearest constellation point (or set soft prob accordingly)
        d = np.abs(constellation - sym)
        k = np.argmin(d)
        q = np.zeros(S); q[k] = 1.0
        qx_probs[idx, :] = q

    # initial q(x) means and second moments
    qx_means = qx_probs.dot(constellation)
    qx_second = (qx_probs * (np.abs(constellation)**2)).sum(axis=1)

    # init q(h) using LS on pilots: h_ls = sum conj(x_p)*y_p / sum |x_p|^2
    if len(pilot_positions) > 0:
        ys = y[pilot_positions]
        xs = pilot_symbols
        h_ls = np.sum(np.conj(xs) * ys) / (np.sum(np.abs(xs)**2) + 1e-12)
        m_h = h_ls
        v_h = 1e-2
    else:
        m_h = 0+0j
        v_h = prior_var_h

    # main iteration
    for it in range(max_iter):
        old_mh = m_h

        # 1) update q(h)
        # use current qx_second
        m_h_new, v_h_new = update_qh(y, qx_means, qx_second, sigma2, prior_var_h)
        # damping
        m_h = damping * m_h_new + (1-damping) * m_h
        v_h = damping * v_h_new + (1-damping) * v_h

        # 2) update q(x_n) for data positions (not pilots)
        for n in range(N):
            if n in pilot_positions:
                continue
            probs = soft_demod_update(y[n], m_h, v_h, sigma2, constellation)
            # damping in probability space (mix old/new)
            probs = damping * probs + (1-damping) * qx_probs[n]
            probs = probs / np.sum(probs)
            qx_probs[n] = probs

        # update moments
        qx_means = qx_probs.dot(constellation)
        qx_second = (qx_probs * (np.abs(constellation)**2)).sum(axis=1)

        # 3) optionally update sigma2
        if estimate_noise:
            sigma2_new = estimate_sigma2(y, m_h, v_h, qx_means, qx_second)
            sigma2 = damping * sigma2_new + (1-damping) * sigma2

        # convergence test (change in m_h)
        if np.abs(m_h - old_mh) < tol:
            if verbose:
                print(f'converged it={it}')
            break

    return {
        'm_h': m_h,
        'v_h': v_h,
        'qx_probs': qx_probs,
        'qx_means': qx_means,
        'sigma2': sigma2,
        'iters': it+1
    }
