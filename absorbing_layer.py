import numpy as np

from numpy.fft import fft, ifft, fftfreq


# create all the stuff we need to implement the absorbing boundary layer


# first, create a function that gives the damping coefficient a la Lu/Trogdon 2023. Delta controls the layer thickness
def damping_coeff(x, length):
    amp = 1.

    l_endpt = -length * 0.5 + 0.5 * length * 0.08

    r_endpt = l_endpt + 0.17 * length

    w = (2 ** -4) * length / 100.

    out = 0.5 * (np.tanh(w * (x - l_endpt)) + 1.) - 0.5 * (np.tanh(w * (x - r_endpt)) + 1.)

    return amp * out


# now, create the Rayleigh damping term that can be added to the forcing
# syntax is inputs is the same as for fourier_forcing
def rayleigh_damping(V, x, length, delta=0.1):
    if int(0.5 * V.size) == x.size:

        pass

    else:

        raise TypeError("The array V must be twice the length of the array x")

    N = int(0.5 * np.size(V))

    V = np.reshape(V, (2 * N,))

    v = np.real(ifft(V[N:]))  # only ifft last N-1 entries of V because of storage conventions

    out = 1j * np.zeros(2 * N, dtype=float)
    beta = damping_coeff(x, length, delta=delta)
    out[N:] = fft(-1. * beta * v)

    return out
