import numpy as np
from numpy.fft import fft, ifft

from joe_main_lib import simulation, model, initial_state, do_refinement_study

# get stgrid
length, T, N, dt = 600., 200., 2 ** 12, 1e-4
# TODO: 600 is so big that cosh(x near endpts) approx infty and we get a RuntimeWarning
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}


# get model
def symbol(k):
    return 1j * k ** 3


def fourier_forcing(V, k, x, nonlinear=True):
    out = 6. * float(nonlinear) * (1j * k) * (0.5 * fft(np.real(ifft(V)) ** 2) - (1. / 3.) * fft(np.real(ifft(V)) ** 3))
    return out


my_model = model('gardner', 1, symbol, fourier_forcing, nonlinear=True)


# get initial state
# note that, for large domains, cosh(x near endpts) may encounter overflow, so it's good to just manually set
# the tails of the wave to zero to avoid these concerns (see also the great discussion here
# https://stackoverflow.com/questions/31889801/overflow-in-numpy-cosh-function
# which has inspired my approach below)
def gardner_soliton(x, c=1., p=1.):
    out = np.zeros_like(x, dtype=float)
    out[abs(x) > 270] = 0.
    out[abs(x) <= 270] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c) * x[abs(x) <= 270]))
    return out

m = 30  # number of solitons in the gas

def soliton_gas_ic(x, m):
    out = 0.

    phases = np.linspace(-0.5*length+10, 0.5*length-10, num=m, endpoint=True)

    amps = np.random.uniform(low=2.3, high=3., size=m)

    cs = (amps-1.)**2-1.  # using the Gardner NL dispersion relation

    for k in range(0, m):
        out += gardner_soliton(x - phases[k], c = cs[k], p=1.)

    return out


my_initial_state = initial_state('soliton_gas', lambda x: soliton_gas_ic(x, m))

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=200)

my_sim.load_or_run(print_runtime=True, save=True)
my_sim.hov_plot(dpi=200, usetex=True, save_figure=True, show_figure=True)
my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

nsteps = int(T / (200 * dt))
t = np.linspace(0, T, num=nsteps + 1, endpoint=True)

my_sim.get_fm()
my_sim.get_sm()
print(np.amax(my_sim.fm_error))
print(np.amax(my_sim.sm_error))