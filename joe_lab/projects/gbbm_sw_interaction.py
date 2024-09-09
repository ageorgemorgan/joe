import time

import numpy as np

from scipy.signal import argrelmin, argrelmax # for postprocessing: amplitude hist and cdf
from scipy.stats import ecdf # again for postprocessing

import joe_lab.joe as joe

from gbbm_utils import *

# fix basic params
ndump = 10

# get stgrid
length, T, N, dt = 100., 240., 2 ** 13, 2e-3
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

# get model
my_model = joe.model('gardner-bbm', 1, gbbm_symbol, gbbm_fourier_forcing, nonlinear=True)

cplus = 2. # speed of + wave
aplus = gbbm_sw_amp(cplus, 1) # amplitude of + wave
aminus = aplus # amplitude of - wave
cminus = -1.+(1.+aminus)**2 # a bit of algebra

def my_initial_state_func(x):
    return gardnerbbm_solitary_wave(x-15, c=cplus, p=1) + gardnerbbm_solitary_wave(x+15, c=cminus, p=-1)

initial_state_kw = 'gbbm_interaction_cplus=%.2f_cminus=%.2f' % (cplus, cminus)
my_initial_state = joe.initial_state(initial_state_kw, my_initial_state_func)

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc ='periodic', ndump=ndump)
my_sim.plot_initial_condition(show_figure=True, save_figure=False)

my_sim.load_or_run(save=True)

# check energy cons
E = energy(my_sim.Udata, length)
E_error = np.amax(np.abs(E-E[0])/E[0])
from decimal import Decimal
print('Relative error in energy = %.3E' % Decimal(E_error))

my_sim.hov_plot(save_figure=True, show_figure=True, umin=-aminus, umax=aplus)
#my_sim.save_movie(dpi=80, fps=200, usetex=False)