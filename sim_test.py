import pickle

import numpy as np

from numpy.fft import fft, ifft

from main_lib import model, initial_state, simulation

from models import builtin_model

from initial_states import builtin_initial_state

import time

# first prescribe all the simulation parameters etc.

T = 25.  # time to stop simulation at

dt = 2**-7  # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid nodes
N = 2 ** 8

# get the model object from the built-in options just using the
model_kw = 'phi4'
my_model = builtin_model(model_kw, nonlinear=True)

"""
# ALTERNATIVELY, define the symbols manually...
def my_symbol(k):
    return 1j * k ** 3


def my_fourier_forcing(V, k, nonlinear):
    p = 1.

    out = -6. * float(nonlinear) * (1. / (p + 1.)) * 1j * k * (fft(np.real(ifft(V)) ** (p + 1)))

    return out

my_model = model(model_kw, 1, my_symbol, my_fourier_forcing, nonlinear=True)
"""

# get the initial state object from the built-in options:
initial_state_kw = 'gaussian_odd'
my_initial_state = builtin_initial_state(initial_state_kw)

"""
# alternatively, put in the initial state manually
# TODO: get rid of the large output dimension syntax! This default is awful
def my_initial_state_func(x):
    out = np.zeros([2, np.size(x)], dtype=float)

    out[0, :] = 6. * np.exp(-x ** 2)

    return out

my_initial_state = initial_state(initial_state_kw, my_initial_state_func)
"""

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions
my_sim = simulation(length, T, N, dt, my_model, my_initial_state, absorbing_layer=False, ndump=6)

# extract the filename attribute
my_filename = my_sim.filename

try:
    # """
    # load the pkl file containing the sim data (if it exists!) to save a lot of time
    with open(my_filename, 'rb') as inp:
        my_sim = pickle.load(inp)

        print('Saved simulation found, loading saved data.')
    # """

except:

    # if the sim has not been saved, run it and save it

    print('No saved simulation found, running simulation.')

    start = time.time()

    my_sim.run_sim(method_kw='etdrk4') #, splitting_method_kw='naive')

    end = time.time()

    runtime = end - start
    print('Simulation runtime = ', runtime, 's')

    my_sim.save()

"""
# report magnitude of last Fourier coefficient
u = my_sim.Udata #[0,:,:] for second order

v = np.absolute(fft(u))# , axis=1)) # MUST ADD LAST BIT FOR SECOND-ORDER

m = int(0.5*N) - 1  # index of largest positive frequency

v_last = np.amax(v) #[:,m]

print('Maximum over time of modulus of last Fourier coefficient at N =', N, 'is = ', v_last)
"""

# produce plots and movies
"""
x = my_sim.x

u = my_sim.Udata[-1]

import matplotlib.pyplot as plt

plt.plot(x,u)

plt.show()
"""

my_sim.hov_plot(colourmap='cmo.haline', show_figure=True, save_figure=False)

# my_sim.save_movie()

# my_sim.save_combomovie()
