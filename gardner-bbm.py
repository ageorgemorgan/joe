import numpy as np

from joe import simulation, initial_state
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 100., 200., 2**9, 1e-2
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

my_model = builtin_model('gardner-bbm', nonlinear=True)
my_initial_state = builtin_initial_state('gardner-bbm_solitary_wave')

c = 1.
p = 1.

def gardnerbbm_solitary_wave(x, c = 1., p = 1.):

    out = np.zeros_like(x, dtype=float)
    xmax = 1e3
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c/(1.+c)) * x[abs(x) <= xmax]))

    return out

my_initial_state = initial_state('gardnerbbm_solitary_wave', lambda x: gardnerbbm_solitary_wave(x,c,p))

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

#my_sim.plot_initial_condition(color='xkcd:cerulean', usetex=True, show_figure=True, save_figure=False)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=False)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=False, usetex=True)
#my_sim.save_movie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# report error in first moment
my_sim.get_fm()
error = np.amax(my_sim.fm_error)
print('Maximum error in first moment = %.2E' % error)