import numpy as np
from joe_lab.utils import dealiased_pow, integrate

# a bunch of helpers for my Gardner-BBM (GBBM) experiments. Just putting them here so I dont have to repeat myself
# across a bunch of different trials.

def gardnerbbm_solitary_wave(x, c = 1., p = 1.):

    out = np.zeros_like(x, dtype=float)
    xmax = 7e2
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c/(1.+c)) * x[abs(x) <= xmax]))

    return out

# get amplitude of solitary wave
def gbbm_sw_amp(c,p):
    return np.abs( c / (-1. + p * np.sqrt(1. + c)) )

def gbbm_symbol(k):
    return 1j * (k ** 3) / (1. + k ** 2)

def gbbm_fourier_forcing(V,k,x,nonlinear=True):

    Fu2 = dealiased_pow(V,2)
    Fu3 = dealiased_pow(V,3)

    out = 6. * float(nonlinear) * ((1j * k) / (1. + k ** 2)) * (
            0.5 * Fu2 - (1. / 3.) * Fu3)

    return out

# define a function that computes the H1 energy
from scipy.fft import rfft, irfft, rfftfreq

def energy(u,length):
    N = np.shape(u)[-1]

    # get wavenumbers for the grid of S^1 with N samples
    k = 2. * np.pi * N * rfftfreq(N) / length

    spring = irfft(1j * k * rfft(u)) ** 2

    out = 0.5*integrate(np.absolute(u)**2 + spring, length)

    return out
