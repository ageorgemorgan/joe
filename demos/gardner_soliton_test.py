import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models


# get stgrid
length, T, N, dt = 100., 20., 2 ** 10, 1e-3
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}


# get model
my_model = joe_models.builtin_model('gardner', nonlinear=True)

# get initial state

def gardner_soliton(x, c=1., p=1.):
    return c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c) * x))


def my_initial_state(x):
    c = 2.

    p = -1.

    out = gardner_soliton(x, c=c, p=p)
    return out


my_initial_state = joe.initial_state('gardner_soliton', my_initial_state)

# now define the simulation object
my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=200)

# run it
method_kw = 'etdrk4'
my_sim.load_or_run(method_kw=method_kw, print_runtime=True, save=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=False, usetex=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

# report error in first and second moments, which should both be zero on paper
my_sim.get_fm()
my_sim.get_sm()
fm_error = np.amax(my_sim.fm_error)
print('Maximum error in first moment = %.2E' % fm_error)
sm_error = np.amax(my_sim.sm_error)
print('Maximum error in second moment = %.2E' % sm_error)

#nmin, nmax = 8, 14
#Ns = np.array([2 ** 9, 2 ** 10])
#dts = np.flip(np.logspace(-nmax, -nmin, num=nmax - nmin + 1, base=2.))

#joe.do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw=method_kw,
#                    show_figure=True, save_figure=True, usetex=True, fit_min=3, fit_max=9)
