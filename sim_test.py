import pickle

import os

import numpy as np

from numpy.fft import fft, ifft

from joe_main_lib import model, initial_state, simulation

from models import builtin_model

from initial_states import builtin_initial_state

import time

# first prescribe all the simulation parameters etc.

T = 150.  # time to stop simulation at

dt = 2**-6 # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 32.*np.pi # 400.

# number of grid nodes
N = 2 ** 7

# get the model object from the built-in options just using the
model_kw = 'ks'
my_model = builtin_model(model_kw, nonlinear=True)

# get the initial state object from the built-in options:
initial_state_kw = 'ks_chaos'
my_initial_state = builtin_initial_state(initial_state_kw)

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions
my_sim = simulation(length, T, N, dt, my_model, my_initial_state, bc='periodic', ndump=10)

# extract the filename attribute
my_filename = my_sim.filename

try:
    # """
    # load the pkl file containing the sim data (if it exists!) to save a lot of time
    with open('sim_archive/'+my_filename, 'rb') as inp:
        my_sim = pickle.load(inp)

        print('Saved simulation found, loading saved data.')
    # """

except:

    # if the sim has not been saved, run it and save it

    print('No saved simulation found, running simulation.')

    my_sim.run_sim(method_kw='etdrk4') #, splitting_method_kw='naive')

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

my_sim.hov_plot(colourmap='cmo.solar', show_figure=True, save_figure=True)

#my_sim.save_movie(dpi=400)
#my_sim.save_combomovie()
