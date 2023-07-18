import pickle

import numpy as np

from numpy.fft import fft, ifft, fftfreq

from scipy.integrate import simpson

import matplotlib.pyplot as plt

from simulation_lib import simulation

from phi4_model import K0, V0

import time

# first prescribe all the simulation parameters etc.

T = 25.  # time to stop simulation at

dt = 2**-7 # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 64.

# number of grid cells per unit axis
N = 2 ** 8

initial_state_kw = 'gaussian_odd'

nonlinear = True

# obtain the relevant simulation, either by loading it up again or by running it.

try:

    filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=' + initial_state_kw + '_nonlinear=' + str(nonlinear) + '.pkl'

    with open(filename, 'rb') as inp:

        my_sim = pickle.load(inp)

except:

    my_sim = simulation(length, T, N, dt, initial_state_kw, nonlinear=nonlinear, absorbing_layer=False)

    my_sim.run_sim()

    my_sim.save()

x = my_sim.x
Udata = my_sim.Udata

my_sim.hov_plot(show_figure=True, save_figure=False)

# my_sim.save_movie()

# now define a function that computes the energy of our state [u,ut] at time t


def energy(u, ut, x=x, N=N, length=length):
    kin = ut ** 2

    # get wavenumbers for the grid of S^1 with N samples
    k = 2. * np.pi * N * fftfreq(N) / length

    uhat = fft(u)

    ux = np.real(ifft(1j * k * uhat))

    spring = ux ** 2

    potential = (2. + V0(x)) * (u ** 2) + 1.*(2. * K0(x) * u ** 3 + 0.5 * u ** 4)

    # do Simpson integration
    out = 0.5 * simpson(kin + spring + potential)

    return out


# now get the energies associated to each time

times = np.linspace(0., T, num=1 + int(nsteps / my_sim.ndump), endpoint=True)

E = np.zeros_like(times, dtype=float)

for n in np.arange(0, 1+int(nsteps/my_sim.ndump)):

    u = Udata[0, n, :]

    ut = Udata[1, n, :]

    E[n] = energy(u, ut)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

plt.plot(times, (E-E[0])/E[0], '-', color='xkcd:teal', linewidth='2')

plt.xlim([0, T])
# plt.ylim([-5.6,2])

plt.xlabel(r"$t$", fontsize=26, color='k')
plt.ylabel(r"Relative Error in $E[u]$", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('energy_test_L_N=2^8_dt=2^-7', bbox_inches='tight', dpi=800)

plt.show()
