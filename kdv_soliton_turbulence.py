import numpy as np
np.random.seed(32)

from joe_main_lib import simulation, initial_state, do_refinement_study
from models import builtin_model


def kdv_soliton(x, c=1.):
    return 0.5*c*(np.cosh(0.5*np.sqrt(c)*(x))**-2)


def soliton_gas_ic(x,m,length):

    out = 0.

    phases = np.linspace(-290, 290, num=30, endpoint=True)

    amps = np.random.uniform(low=1., high=2., size=m)

    cs = 2.*amps

    for k in range(0,m):

        out += kdv_soliton(x-phases[k], c=cs[k])

    return out

length, T, N, dt = 600., 100., 2**12, 5e-4
m = 30  # number of solitons in the gas

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('kdv', nonlinear=True)
my_initial_state = initial_state('soliton_gas', lambda x: soliton_gas_ic(x, m, length))

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=200)

x = my_sim.x

my_sim.run_sim(print_runtime=True)
my_sim.hov_plot(usetex=True, save_figure=True, show_figure=True)
my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
