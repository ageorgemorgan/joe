import numpy as np

from numpy.fft import fft, ifft, fftfreq

from scipy import sparse


# Aux functions needed for special cases...

# obtain the kink
def K0(x):
    out = np.tanh(x / np.sqrt(2))

    return out


# obtain the potential associated to the kink
# Note: the +2 in the potential gets put into linear part of evolution eq.
def V0(x):
    out = -3. * np.cosh(x / np.sqrt(2)) ** -2

    return out


# Here begins the actual core of the material

# if model is first order in time, below just gives linear, const. coeff. part. in Fourier space
# if model is second order in time, instead obtain the spatial operator for the first order system as a block matrix
def get_spatial_operator(length, N, model_kw):
    # get wavenumbers for the grid of S^1 with N samples
    k = 2 * np.pi * N * fftfreq(N) / length

    if model_kw == 'phi4':
        # linear, constant-coefficient part of PDE
        L = -(k ** 2 + 2. * np.ones_like(k))  # CHANGE FOR WAVE EQN

        # put L together into sparse block matrix , multiply by dt
        A = sparse.diags([L, np.ones(N, dtype=float)], [-N, N], shape=[2 * N, 2 * N]).tocsc()

    elif model_kw == 'bbm':
        A = 1j * (k**3) / (1. + k ** 2)     # -1j * k / (1. + k ** 2)

    elif model_kw == 'kdv':
        A = 1j*k**3

    elif model_kw == 'shore_kdv':
        A = -1j*(k-k**3)

    elif model_kw == 'kawahara':
        A = -1j * (-(9./20.)*k + k ** 3 - k**5)  # Note how this is the Kawahara dispersion in the frame travelling
        # with the head of the wave train (at a group vel of c_g = 9/20)

    return A


def fourier_forcing(V, x, length, model_kw, nonlinear=True):
    # Fourier transform of forcing term, acting on pair fncs V=(v_1, v_2)^T (concatenation)
    # on Fourier space. V has size 2N

    if model_kw == 'phi4':

        if int(0.5 * V.size) == x.size:

            pass

        else:

            raise TypeError("The array V must be twice the length of the array x")

        N = int(0.5 * np.size(V))

        V = np.reshape(V, (2 * N,))

        u = np.real(ifft(V[0:N]))  # only ifft first N entries of V because of storage conventions

        spatial_forcing = -1. * V0(x) * u - float(nonlinear) * (3. * K0(x) * u ** 2 + u ** 3)

        out = 1j * np.zeros(2 * N, dtype=float)
        out[N:] = fft(spatial_forcing)

    elif model_kw == 'bbm':
        N = np.size(V)

        k = 2 * np.pi * N * fftfreq(N) / length

        p = 1.

        out = -6.*float(nonlinear)*(1. / (p + 1.)) * 1j * k / (1. + k ** 2) * (fft(np.real(ifft(V)) ** (p + 1)))

    elif model_kw == 'kdv' or 'kawahara' or 'shore_kdv':
        N = np.size(V)

        k = 2 * np.pi * N * fftfreq(N) / length

        p = 1.

        out = -6.*float(nonlinear) * (1. / (p + 1.)) * 1j*k * (fft(np.real(ifft(V)) ** (p + 1)))

    return out
