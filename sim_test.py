from joe_main_lib import model, initial_state, simulation

from models import builtin_model

from initial_states import builtin_initial_state

# first prescribe all the simulation parameters etc.

T = 150.  # time to stop simulation at

dt = 0.01 # time step size

nsteps = int(T / dt)  # total num of time steps we take

length = 400.

# number of grid nodes
N = 2 ** 10

# space-time grid
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}

# get the model object from the built-in options just using the
model_kw = 'kdv'
my_model = builtin_model(model_kw, nonlinear=True)

# get the initial state object from the built-in options:
initial_state_kw = 'gaussian_even_alt'
my_initial_state = builtin_initial_state(initial_state_kw)

# create the simulation object by prescribing physical parameters, discretization parameters, initial conditions
my_sim = simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', ndump=20)

# extract the filename attribute
my_filename = my_sim.filename

my_sim.load_or_run(method_kw='ifrk4', print_runtime=True, save=False)

"""
# report magnitude of last Fourier coefficient
u = my_sim.Udata #[0,:,:] for second order

v = np.absolute(fft(u))# , axis=1)) # MUST ADD LAST BIT FOR SECOND-ORDER

m = int(0.5*N) - 1  # index of largest positive frequency

v_last = np.amax(v) #[:,m]

print('Maximum over time of modulus of last Fourier coefficient at N =', N, 'is = ', v_last)
"""

# produce plots and movies
"""
x = my_sim.x

u = my_sim.Udata[-1]

import matplotlib.pyplot as plt

plt.plot(x,u)

plt.show()
"""

# produce plots and movies
my_sim.hov_plot(colormap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=False)
#my_sim.save_movie(dpi=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')
#my_sim.save_combomovie(dpi=200, usetex=False, fieldcolor='xkcd:cerulean', speccolor='xkcd:dark magenta', fieldname='u')

# do refinement study to verify accuracy
#nmin, nmax = 1, 9
#Ns = np.array([2**6, 2**7, 2**8])
#dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
#do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='sponge_layer', show_figure=True, save_figure=True, usetex=True)
