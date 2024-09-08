import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

length, T, N, dt = 100., 100., 2**10, 1e-2
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

my_model = joe_models.builtin_model('kdv', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('kdv_soliton')

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

#my_sim.plot_initial_condition(usetex=True, show_figure=True, save_figure=True)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save = True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# report error in first and second moments, which should both be zero on paper
my_sim.get_fm()
my_sim.get_sm()
fm_error = np.amax(my_sim.fm_error)
print('Maximum error in first moment = %.2E' % fm_error)
sm_error = np.amax(my_sim.sm_error)
print('Maximum error in second moment = %.2E' % sm_error)
