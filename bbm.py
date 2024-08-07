import numpy as np

from joe_main_lib import simulation, do_refinement_study, do_refinement_study_alt
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 100., 20., 2**10, 1e-3

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('bbm', nonlinear=True)
my_initial_state = builtin_initial_state('bbm_solitary_wave')

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=True)

# produce plots and movies
my_sim.hov_plot(colormap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')
