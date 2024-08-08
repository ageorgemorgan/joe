import numpy as np
np.random.seed(32)

from joe_main_lib import simulation, initial_state, do_refinement_study
from models import builtin_model


def bbm_solitary_wave(x,c=1.):
    return 0.5*c*(np.cosh(0.5*np.sqrt(c/(1.+c))*(x))**-2)


def sample_one_phase(length):
    return np.random.uniform(low=-0.5 * length, high=0.5 * length)


def get_sample_phases(m, length, min_dist=20.):
    x = np.zeros(m)

    x[0] = sample_one_phase(length)

    for k in range(1, m):

        draw_new_sample = True

        while draw_new_sample:

            y = sample_one_phase(length)

            ds = np.array([np.abs(y - xx) >= min_dist for xx in x[0:k]])

            bdry_ds = np.array([np.abs(y - xx) >= 0.5*min_dist for xx in np.array([-0.5*length, 0.5*length])])

            if ds.all() and bdry_ds.all():

                draw_new_sample = False

        x[k] = y

    return x

def soliton_gas_ic(x,m,length):

    out = 0.

    phases = get_sample_phases(m, length, min_dist=0.033*length)

    mm = int(0.5*m)

    amps_plus = np.random.uniform(low=1., high=2., size=mm)

    amps_minus = -1.*np.random.uniform(low=2., high=3., size=mm)

    amps = np.concatenate((amps_plus, amps_minus))

    #amps = np.random.uniform(low=1., high=2., size=m)

    cs = 2.*amps

    for k in range(0,m):

        out += bbm_solitary_wave(x-phases[k], c=cs[k])

    return out

length, T, N, dt = 600., 100., 2**10, 1e-3
m = 2 # number of solitons in the gas

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('bbm', nonlinear=True)
my_initial_state = initial_state('soliton_gas', lambda x: soliton_gas_ic(x, m, length))

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=200)

#import matplotlib.pyplot as plt
#x = my_sim.x
#plt.plot(x, soliton_gas_ic(x,m,length))
#plt.show()

#my_sim.run_sim(print_runtime=True)
#my_sim.hov_plot(usetex=True, save_figure=False, show_figure=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

Ns = np.array([2**10])
dts = np.array([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw='etdrk4', show_figure=True,
                        save_figure=True, usetex=True, fit_min=0, fit_max=4)