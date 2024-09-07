import numpy as np
import joe_lab.utils as joe_utils

def test_dealiased_pow():
    N,k,p = 32, 2, 2
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    u = np.sin(k * x)
    V = joe_utils.my_fft(u, complex=False)

    out_naive = joe_utils.my_fft(joe_utils.my_ifft(V)**p)
    out_dealiased = joe_utils.dealiased_pow(V,p)

    assert np.amax(np.abs(out_naive-out_dealiased))<1e-12