# Demo: solving the Kuramoto-Sivashinsky equation
# (after Kassam & Trefethen 2005: https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf)
import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

length, T, N, dt = 32.*np.pi, 150., 2**7, 2**-6
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}
my_model = joe_models.builtin_model('ks', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('ks_chaos')
my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=10)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=False)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.solar', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:dark orange', fieldname='u')
#my_sim.save_combomovie(dpi=200, usetex=False, fieldcolor='xkcd:dark orange', speccolor='xkcd:dark magenta', fieldname='u')

# report error in first moment
my_sim.get_fm()
error = np.amax(my_sim.fm_error)
print('Maximum error in first moment = %.2E' % error)

# do refinement study to verify accuracy
#nmin, nmax = 1, 9
#Ns = np.array([2**6, 2**7, 2**8])
#dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
#do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', show_figure=True, save_figure=True, usetex=True)
