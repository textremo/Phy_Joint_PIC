import numpy as np

def linear_convolution_fft(x, h):
    N = len(x) + len(h) - 1
    X = np.fft.fft(x, N)
    H = np.fft.fft(h, N)
    Y = X * H
    y = np.fft.ifft(Y)
    return np.real(y)  # 理论上是实数，可以取实部

# 示例
x = [1, 2, 3]
h = [4, 5]

y = linear_convolution_fft(x, h)
print(y)