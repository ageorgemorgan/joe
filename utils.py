import numpy as np
from scipy.fft import fft, ifft, rfft, irfft


def my_fft(u, n=None, complex=False):
    r"""Takes FFT of a real or complex array along the -1 axis, optimizing storage if the array is real.

    Wraps the scipy.fft routines fft, rfft.

    Parameters
    ----------
        u : ndarray
            Real or complex numpy array.
        n: int or None, optional.
            Total number of frequencies to include, if padding. Default: None (no padding).
        complex : boolean, optional
            Equal to True if u is complex and False if u is real. Default: False.

    Returns
    -------
        out : complex ndarray
            Same shape conventions as scipy.fft.fft <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#scipy.fft.fft>
            and scipy.fft.rfft <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html#scipy.fft.rfft>.
    """
    if complex:

        out = fft(u, n=n)

    else:

        out = rfft(u, n=n)

    return out

def my_ifft(V, n=None, complex=False):
    r"""Takes inverse FFT of an array along the -1 axis, optimizing storage if the inverse FFT is expected to be real.

        Wraps the scipy.fft routines ifft, irfft.

        Parameters
        ----------
            V : complex ndarray
                Array to inverse FFT
            n: int or None, optional.
                Total number of frequencies to pad V with, if padding is desired. Default: None (no padding).
            complex : boolean, optional
                Equal to True if inverse FFT of V is known to be complex and False if inverse FFT is known to be
                real. Default: False.

        Returns
        -------
            out: ndarray
                Same shape conventions as scipy.fft.ifft <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html#scipy.fft.ifft>
                and scipy.fft.irfft <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.rfft.html#scipy.fft.irfft>.
        """
    if complex:

        out = ifft(V, n=n)

    else:

        out = irfft(V, n=n)

    return out

def integrate(u, length):
    r"""Integrates N samples of a real or complex space-time field over the spatial interval [-0.5*length, 0.5*length]
       using the FFT.

        Parameters
        ----------
            u : ndarray
                Real or complex array to be integrated in space. Convention is that spatial variation is stored in the
                -1 axis.
            length : float
                Total length of spatial domain.

        Returns
        -------
            out: float or ndarray
                Approximation of the integral of the sampled space-time field over the spatial interval
                [-0.5*length, 0.5*length].
    """
    N = np.shape(u)[-1]
    return (length/N) * np.real(fft(u, axis=-1)[..., 0])

def dealiased_pow(V,p):
    r"""Compute the Fourier-space version of a power of an array, dealiased by zero-padding.

        Important: this only works for REAL fields because for complex fields :math:`u`, algebraic nonlinearities
        typically involve the complex conjugate :math:`\overline{u}` as well as :mathm:`u`. So, padding for such
        nonlinearities terms should be done on-the-fly. A future release of *joe* may include support for complex
        powers.

        Parameters
        ----------
            V : complex ndarray

            p : int

        Returns
        -------
            out: complex ndarray
                A version of rfft(irfft(V)**p) that has been dealiased via zero-padding
    """
    #

    N = 2 * (len(V) - 1)
    K = int(0.5*(p+1)*N)
    upad = my_ifft(V, n=K, complex=False)

    # FOR CLARITY/DEV. PURPOSES ONLY: the above lines of code produce the same result as the following block:
    #Vpad = 1j * np.zeros(int(0.5 * K + 1), dtype=float)
    #Vpad[0:len(V)] = V
    #upad = irfft(Vpad, n=K)

    out = ((K/N)**(p-1))*my_fft(upad ** p, complex=False)[0:int(0.5*N+1)]

    # TODO: I discovered the correct normalizations via trial-and error, and by comparing Fu2, Fu3 against
    #  rfft(irfft(V) ** 2), rfft(irfft(V) ** 3) resp. Make sure to write in a tutorial, or in the docs, a more
    #  systematic way of determining this normalization. Dealiasing via padding would be a great topic for a
    #  "tutorial 5"! U could also experiment with filtering vs. padding vs. doing nothing!

    return out
