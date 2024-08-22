import numpy as np
from scipy.fft import fft, ifft, rfft, irfft

# helper function for efficiently taking fft of real or complex fields
def my_fft(u, complex=False):

    if complex:

        out = fft(u)

    else:

        out = rfft(u)

    return out

def my_ifft(V, complex=False):

    if complex:

        out = ifft(V)

    else:

        out = irfft(V)

    return out

# helper function for integration (of real part of fnc) in space. Uses FFT to accurately integrate over spatial domain:
# accuracy vastly beats trapezoidal rule.
# u = array storing node values of field to be integrated (last dimension of the array is spatial)
# length = length of domain
# N = number of samples of u we take (= number of grid pts)
# since this is a postprocessing func it doesn't need to be optimized for real inputs with rfft.
def integrate(u, length, N):
    return (length/N) * np.real(fft(u, axis=-1)[..., 0])
