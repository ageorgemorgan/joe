# Demo: solving the KdV equation with a sponge layer
# (after Liu and Trogdon 2023: https://www.sciencedirect.com/science/article/pii/S0168927423001526 )

import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

length, T, N, dt = 400., 150., 2**10, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = joe_models.builtin_model('kdv', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('gaussian_even_alt')  # also try 'kdv_soliton', 'kdv_multisoliton'

# we need a sponge layer here, which requires parameter tuning to get right.
# in joe, the sponge layer params are contained in a dict called "sponge_params"
# that you pass into the init of your simulation.

l_endpt = -length * 0.5 + 0.5 * length * 0.1
r_endpt = l_endpt + 0.01 * length
width = (2 ** -6) * length / 100.
sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                 'width': width, 'expdamp_freq': 1000,
                 'damping_amplitude': 10.,
                 'splitting_method_kw': 'naive',
                 'spongeless_frac': .5}  # this is the fraction of the middle of the spatial domain to keep in the plots

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params, ndump=20)

#my_sim.plot_initial_condition(color='xkcd:cerulean', usetex=True, show_figure=True, save_figure=True)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=False)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, fps=100, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# perform refinement study if desired
#nmin, nmax = 2, 14
#Ns = np.array([2**10])
#dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
#joe.do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='sponge_layer',
#                        sponge_params=sponge_params, method_kw='etdrk4', show_figure=True,
#                        save_figure=True, usetex=True, fit_min=2, fit_max=6)
