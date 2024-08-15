import numpy as np

from joe_main_lib import simulation, do_refinement_study, do_refinement_study_alt
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 400.,150., 2**10, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('phi4', nonlinear=True)
my_initial_state = builtin_initial_state('gaussian_odd')

# we need a sponge layer here, which requires parameter tuning to get right.

l_endpt = -0.5*length  + 0.5 * length * 0.1
r_endpt = l_endpt + 0.05 * length
width = (2 ** -4) * length / 100.
sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                 'width': width, 'expdamp_freq': 1e3,
                 'damping_amplitude': 10.,
                 'spongeless_frac': 0.5}  # this is the fraction of the middle of the spatial domain to keep in the plots

my_sim = simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params, ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save_pkl=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

"""
nmin, nmax = 4, 9
Ns = np.array([2**9, 2**10, 2**11])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))

do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw='etdrk4',
                    show_figure=True, save_figure=True, usetex=True, fit_min=0, fit_max=4)
"""