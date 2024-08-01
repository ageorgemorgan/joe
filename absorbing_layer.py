import numpy as np

from numpy.fft import fft, ifft, fftfreq


# create all the stuff we need to implement the absorbing boundary layer

# first, create a function that gives the damping coefficient a la Lu/Trogdon 2023.
def damping_coeff_lt(x, sponge_params):
    amp = 1.

    l_endpt = sponge_params['l_endpt']  # -length * 0.5 + 0.5 * length * 0.1

    r_endpt = sponge_params['r_endpt']  # l_endpt + 0.01 * length

    w = sponge_params['width']  # (2 ** -6) * length / 100.

    out = 0.5 * (np.tanh(w * (x - l_endpt)) + 1.) - 0.5 * (np.tanh(w * (x - r_endpt)) + 1.)

    return amp * out


# create a function that gives the damping coefficient a la Bronski 1998.
def damping_coeff_bronski(x, length, delta=0.1):
    # left endpoint
    lep = -0.5 * length

    # right endpoint
    rep = 0.5 * length

    condlist = [((lep + delta <= x) & (x <= rep - delta)), ((lep <= x) & (x < lep + delta)),
                ((rep - delta < x) & (x <= rep))]

    w = np.pi / (2. * delta)

    funclist = [lambda x: 0, lambda x: 2. * np.cos(w * (x - lep)), lambda x: 2. * np.cos(w * (rep - x))]

    out = np.piecewise(x, condlist, funclist)

    return out


# create the Rayleigh damping term that can be added to the forcing
# syntax is inputs is the same as that for fourier_forcing
def rayleigh_damping(V, x, length, delta=0.1):
    if int(0.5 * V.size) == x.size:

        pass

    else:

        raise TypeError("The array V must be twice the length of the array x")

    N = int(0.5 * np.size(V))

    V = np.reshape(V, (2 * N,))

    v = np.real(ifft(V[N:]))  # only ifft last N-1 entries of V because of storage conventions

    out = 1j * np.zeros(2 * N, dtype=float)
    beta = damping_coeff_bronski(x, length, delta=delta)
    out[N:] = fft(-1. * beta * v)

    return out


# helper function to clip the "spongeless" part of an array
def clip_spongeless(z, sfrac):
    delta = 0.5 * (1. - sfrac)
    N = np.shape(z)[-1]
    out = z[..., int(delta * N):int((1. - delta) * N) + 1]
    return out
