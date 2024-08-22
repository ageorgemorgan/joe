import numpy as np

from numpy.fft import fft, ifft

import joe
from joe import model


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
def get_symbol(k, model_kw):

    if model_kw == 'phi4pert':
        A = -(k ** 2 + 2. * np.ones_like(k))

    elif model_kw =='sinegordon':

        A = -k**2

    elif model_kw == 'bbm' or model_kw == 'gardner-bbm':
        A = 1j * (k ** 3) / (1. + k ** 2)

    elif model_kw == 'ks':
        A = k ** 2 - k ** 4

    elif model_kw == 'kdv' or model_kw == 'gardner':
        A = 1j * k ** 3

    else:

        raise NameError("Invalid model keyword string.")

    return A


def fourier_forcing(V, k, x, model_kw, nonlinear=True):
    # Fourier transform of forcing term, acting on pair fncs V=(v_1, v_2)^T (concatenation)
    # on Fourier space. V has size 2N

    if model_kw == 'phi4pert':

        if int(0.5 * V.size) == x.size:

            pass

        else:

            raise TypeError("The array V must be twice the length of the array x.")

        N = int(0.5 * np.size(V))

        V = np.reshape(V, (2 * N,))

        u = np.real(ifft(V[0:N]))  # only ifft first N entries of V because of storage conventions

        spatial_forcing = -1. * V0(x) * u - float(nonlinear) * (3. * K0(x) * u ** 2 + u ** 3)

        out = 1j * np.zeros(2 * N, dtype=float)
        out[N:] = fft(spatial_forcing)

    elif model_kw == 'sinegordon':

        if int(0.5 * V.size) == x.size:

            pass

        else:

            raise TypeError("The array V must be twice the length of the array x.")

        N = int(0.5 * np.size(V))

        V = np.reshape(V, (2 * N,))

        u = np.real(ifft(V[0:N]))  # only ifft first N entries of V because of storage conventions

        spatial_forcing = -float(nonlinear)*np.sin(u)

        out = 1j * np.zeros(2 * N, dtype=float)
        out[N:] = fft(spatial_forcing)

    elif model_kw == 'bbm':

        p = 1.

        out = -6. * float(nonlinear) * (1. / (p + 1.)) * 1j * k / (1. + k ** 2) * (fft(np.real(ifft(V)) ** (p + 1)))

    elif model_kw == 'ks':

        p = 1.

        out = -float(nonlinear) * (1. / (p + 1.)) * 1j * k * (fft(np.real(ifft(V)) ** (p + 1)))

    elif model_kw == 'gardner':

        out = 6. * float(nonlinear) * (1j * k) * (
                    0.5 * fft(np.real(ifft(V)) ** 2) - (1. / 3.) * fft(np.real(ifft(V)) ** 3))

    elif model_kw == 'gardner-bbm':

        out =  6. * float(nonlinear) * ((1j * k)/(1. + k**2)) * (
                    0.5 * fft(np.real(ifft(V)) ** 2) - (1. / 3.) * fft(np.real(ifft(V)) ** 3))

    elif model_kw == 'kdv':

        p = 1.

        out = -6. * float(nonlinear) * (1. / (p + 1.)) * 1j * k * (fft(np.real(ifft(V)) ** (p + 1)))

    else:

        raise NameError("Invalid model keyword string.")

    return out


def builtin_model(model_kw, nonlinear=True):

    def my_symbol(k):
        return get_symbol(k, model_kw)

    if model_kw == 'phi4pert' or model_kw == 'sinegordon':

        t_ord = 2

        def my_fourier_forcing(V, k, x, nonlinear):
            return fourier_forcing(V, k, x, model_kw, nonlinear)

    else:

        t_ord = 1

        def my_fourier_forcing(V, k, x, nonlinear):
            return fourier_forcing(V, k, x, model_kw, nonlinear)

    out = joe.model(model_kw, t_ord, my_symbol, my_fourier_forcing, nonlinear=nonlinear)

    return out
