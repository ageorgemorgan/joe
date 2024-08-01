# Demo: solving the Kuramoto-Sivashinsky equation
# (after Kassam & Trefethen 2005: https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf)
import numpy as np

from joe_main_lib import simulation, do_refinement_study
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 32.*np.pi, 150., 2**7, 2**-6
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}
my_model = builtin_model('ks', nonlinear=True)
my_initial_state = builtin_initial_state('ks_chaos')
my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=10)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True)

# produce plots and movies
my_sim.hov_plot(colormap='cmo.solar', fieldname='u', show_figure=True, save_figure=True, usetex=True)
my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:dark orange', fieldname='u')
#my_sim.save_combomovie(dpi=200, usetex=False, fieldcolor='xkcd:dark orange', speccolor='xkcd:dark magenta', fieldname='u')

# do refinement study to verify accuracy
nmin, nmax = 1, 9
Ns = np.array([2**6, 2**7, 2**8])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', show_figure=True, save_figure=True, usetex=True)