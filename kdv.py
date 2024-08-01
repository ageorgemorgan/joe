# Demo: solving the KdV equation with a sponge layer
# (after Liu and Trogdon 2023: https://www.sciencedirect.com/science/article/pii/S0168927423001526 )

from joe_main_lib import simulation, do_refinement_study
from models import builtin_model
from initial_states import builtin_initial_state

length, T, N, dt = 400., 150., 2**10, 0.01

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('kdv', nonlinear=True)
my_initial_state = builtin_initial_state('gaussian_even_alt')

# we need a sponge layer here, which requires parameter tuning to get right.
# in joe, the sponge layer params are contained in a dict called "sponge_params"
# that you pass into the init of your simulation.

l_endpt = -length * 0.5 + 0.5 * length * 0.1
r_endpt = l_endpt + 0.01 * length
width = (2 ** -6) * length / 100.
sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                 'width': width, 'expdamp_freq': 100,
                 'damping_amplitude': 10.,
                 'splitting_method_kw': 'naive',
                 'spongeless_frac': 0.5}  # this is the fraction of the middle of the spatial domain to keep in the plots

my_sim = simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params, ndump=20)

my_sim.load_or_run(method_kw='ifrk4', print_runtime=True, save=True)

# produce plots and movies
my_sim.hov_plot(colormap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
my_sim.save_movie(dpi=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
my_sim.save_combomovie(dpi=200, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# do refinement study to verify accuracy
#Ns = np.array([2**9, 2**10, 2**11])
#dts = np.array([0.05, 0.025, 0.01, 0.0075])
#do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='sponge_layer', method_kw='ifrk4',
#                  show_figure=True, save_figure=True, usetex=True, fit_min=1, fit_max=3)