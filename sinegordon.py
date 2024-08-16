import numpy as np

from joe_main_lib import simulation, do_refinement_study
from models import builtin_model
from initial_states import builtin_initial_state, sinegordon_soliton

length, T, N, dt = 100.,70., 2**9, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('sinegordon', nonlinear=True)
my_initial_state = builtin_initial_state('sinegordon_soliton_interaction_alt') #sinegordon_soliton_interaction

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save_pkl=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.dense', fieldname='u', show_figure=True, save_figure=True, usetex=True)
my_sim.save_movie(dpi=200, fps=90, usetex=False, fieldcolor='xkcd:deep magenta', fieldname='u')

"""
nmin, nmax = 4, 9
Ns = np.array([2**9, 2**10, 2**11])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))

do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw='etdrk4',
                    show_figure=True, save_figure=True, usetex=True, fit_min=0, fit_max=4)
"""