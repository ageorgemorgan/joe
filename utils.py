import numpy as np
from scipy.fft import fft, ifft, rfft, irfft

# helper function for efficiently taking fft of real or complex fields
def my_fft(u, n=None, complex=False):

    if complex:

        out = fft(u, n=n)

    else:

        out = rfft(u, n=n)

    return out

def my_ifft(V, n=None, complex=False):

    if complex:

        out = ifft(V, n=n)

    else:

        out = irfft(V, n=n)

    return out

# helper function for integration (of real part of fnc) in space. Uses FFT to accurately integrate over spatial domain:
# accuracy vastly beats trapezoidal rule.
# u = array storing node values of field to be integrated (last dimension of the array is spatial)
# length = length of domain
# N = number of samples of u we take (= number of grid pts)
# since this is a postprocessing func it doesn't need to be optimized for real inputs with rfft.
def integrate(u, length, N):
    return (length/N) * np.real(fft(u, axis=-1)[..., 0])

# symbol of the lo-pass Orszag a-filter with 0<a<1 (typically a=2/3). Ref: Boyd Ch.11
def orszag(k, length, N, a=2./3.):
    km = np.pi*N/length # max freq
    akm = a*km

    # option a: classical hard transition
    #out = np.zeros_like(k, dtype=float)
    #out[abs(k) <= akm] = 1.
    #out[abs(k) >= akm] = 0.

    # option b: softened transition
    W = 1e2
    out = 0.5*(1.-np.tanh(W*(k-akm)))

    return out

def dealiased_pow(V,p):
    # returns a version of rfft(irfft(V)**p) dealiased via zero-padding

    # Important: this fnc only works for REAL fields bcz for complex fields algebraic nonlinearities typically involve
    # u* as well as u... so padding for such NL terms should be done on-the-fly.

    N = 2 * (len(V) - 1)

    K = int(0.5*(p+1)*N)

    upad = my_ifft(V, n=K, complex=False)

    # FOR DOC. PURPOSES ONLY: the above lines of code produce the same result as the following block:
    #Vpad = 1j * np.zeros(int(0.5 * K + 1), dtype=float)
    #Vpad[0:len(V)] = V
    #upad = irfft(Vpad, n=K)

    out = ((K/N)**(p-1))*my_fft(upad ** p, complex=False)[0:int(0.5*N+1)]

    # TODO: I discovered the correct normalizations via trial-and error, and by comparing Fu2, Fu3 against
    #  rfft(irfft(V) ** 2), rfft(irfft(V) ** 3) resp. Make sure to write in a tutorial, or in the docs, a more
    #  systematic way of determining this normalization. Dealiasing via padding would be a great topic for a
    #  "tutorial 5"! U could also experiment with filtering vs. padding vs. doing nothing!

    return out


