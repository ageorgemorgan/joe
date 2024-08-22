import numpy as np

from .utils import my_fft, my_ifft

# create all the stuff we need to implement the sponge layer (absorbing layer/segment near bdry where
# artifical damping turns on)

# first, create a function that gives the damping coefficient a la Lu/Trogdon 2023.
# TODO: should this be made to work on both sides of the domain rather than just one?
def damping_coeff_lt(x, sponge_params):
    amp = 1.

    l_endpt = sponge_params['l_endpt']  # -length * 0.5 + 0.5 * length * 0.1

    r_endpt = sponge_params['r_endpt']  # l_endpt + 0.01 * length

    w = sponge_params['width']  # (2 ** -6) * length / 100.

    out = 0.5 * (np.tanh(w * (x - l_endpt)) + 1.) - 0.5 * (np.tanh(w * (x - r_endpt)) + 1.)

    return amp * out


# create a function that gives the damping coefficient a la Bronski 1998.
# TODO: update this! needs to play nicely with sponge params.
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
def rayleigh_damping(V, x, length, sponge_params, complex=False):

    N = x.size

    if int(V.size - 2) == N or int(0.5*V.size) == N:

        pass

    else:

        raise TypeError("The array V must be 2+(size of the array x) if our soln is real, "
                        "or 2*(size of the array x) if our soln is complex."
                        " Size of V = ", int(V.size), "size of x = ", x.size)

    if complex:

        NN = N

    else:

        NN = int(0.5*N)+1

    V = np.reshape(V, (2*NN))

    v = my_ifft(V[NN:], complex=complex)  # only ifft last NN entries of V because of storage conventions

    beta = damping_coeff_lt(x, sponge_params)+damping_coeff_lt(-x, sponge_params)
    out = 1j * np.zeros(int(2*NN), dtype=float)
    out[NN:] = my_fft(-1. * beta * v, complex=complex)

    return out


# helper function to clip the "spongeless" part of an array
def clip_spongeless(z, sfrac):
    delta = 0.5 * (1. - sfrac)
    N = np.shape(z)[-1]
    out = z[..., int(delta * N):int((1. - delta) * N) + 1]
    return out
