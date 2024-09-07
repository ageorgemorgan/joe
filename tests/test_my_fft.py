import numpy as np
import joe_lab.utils as joe_utils

N = 32

def test_my_fft_r():
    u = np.zeros(N, dtype=float)
    V = joe_utils.my_fft(u, complex=False)
    assert np.amax(np.abs(V))<1e-16 and len(V) == int(0.5*N+1)

def test_my_fft_c():
    u = 1j*np.zeros(N, dtype=float)
    V = joe_utils.my_fft(u, complex=True)
    assert np.amax(np.abs(V))<1e-16 and len(V) == N
