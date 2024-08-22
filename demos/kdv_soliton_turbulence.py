import numpy as np

np.random.seed(32)

import joe_lab.joe as joe
import joe_lab.models as joe_models


def kdv_soliton(x, c=1.):
    return 0.5 * c * (np.cosh(0.5 * np.sqrt(c) * (x)) ** -2)


length, T, N, dt = 600., 2., 2 ** 12, 1e-4
m = 30  # number of solitons in the gas

def soliton_gas_ic(x, m):
    out = 0.

    phases = np.linspace(-0.5*length+10, 0.5*length-10, num=m, endpoint=True)

    amps = np.random.uniform(low=1., high=2., size=m)

    cs = 2. * amps

    for k in range(0, m):
        out += kdv_soliton(x - phases[k], c=cs[k])

    return out

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = joe_models.builtin_model('kdv', nonlinear=True)
my_initial_state = joe.initial_state('soliton_gas', lambda x: soliton_gas_ic(x, m))

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=200)

my_sim.load_or_run(print_runtime=True, save=True)
my_sim.hov_plot(dpi=600, usetex=True, save_figure=True, show_figure=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

nsteps = int(T / (200 * dt))
t = np.linspace(0, T, num=nsteps + 1, endpoint=True)

my_sim.get_fm()
my_sim.get_sm()
print(np.amax(my_sim.fm_error))
print(np.amax(my_sim.sm_error))
