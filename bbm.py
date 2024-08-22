import numpy as np

from joe import simulation
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 100., 50., 2**10, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('bbm', nonlinear=True)
my_initial_state = builtin_initial_state('bbm_solitary_wave')

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.plot_initial_condition(color='xkcd:cerulean', usetex=True, show_figure=True, save_figure=True)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=False)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# report error in first moment
my_sim.get_fm()
error = np.amax(my_sim.fm_error)
print('Maximum error in first moment = %.2E' % error)
