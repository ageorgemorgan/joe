import pickle

import numpy as np
from numpy.fft import fft

from simulation_lib import simulation

import time

# first prescribe all the simulation parameters etc.

T = 50.  # time to stop simulation at

dt = 2**-9 # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 400.

# number of grid nodes
N = 2 ** 10
model_kw = 'kdv'

initial_state_kw = 'kdv_soliton'

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions, and
# whether or not we want to include nonlinearity
my_sim = simulation(length, T, N, dt, model_kw, initial_state_kw, nonlinear=True, absorbing_layer=True, ndump=20)

# extract the filename attribute
my_filename = my_sim.filename

try:
    #"""
    # load the pkl file containing the sim data (if it exists!) to save a lot of time
    with open(my_filename, 'rb') as inp:
            my_sim = pickle.load(inp)

            print('Saved simulation found, loading saved data.')
    #"""

except:

    # if the sim has not been saved, run it and save it

    print('No saved simulation found, running simulation.')

    start = time.time()

    my_sim.run_sim()

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

my_sim.hov_plot(show_figure=True, save_figure=True)

my_sim.save_movie()

#my_sim.save_combomovie()
