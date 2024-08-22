import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states
from joe_lab.visualization import nice_multiplot
from joe_lab.utils import integrate


length, T, N, dt = 240.,2e4, 2**9, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = joe_models.builtin_model('phi4pert', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('internal_mode')

# we need a sponge layer here, which requires parameter tuning to get right.

l_endpt = -0.5*length  + 0.5 * length * 0.05
r_endpt = l_endpt + 0.05 * length
width = (2 ** -4) * length / 100.
sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                 'width': width, 'expdamp_freq': 1e3,
                 'damping_amplitude': 10.,
                 'spongeless_frac': 0.5}  # this is the fraction of the middle of the spatial domain to keep in the plots

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params, ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=True)

# do data analysis on the amplitudes

# helper funcs first

from joe_lab.sponge_layer import clip_spongeless

def internal_mode(x):

    out = np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    return out

# compute the L2 product of two real arrays via the FFT (sampled funcs)
def L2prod(u1, u2, length=length, N=N):

    M1 = u1.size

    M2 = u2.size

    if M1 == M2:

        pass

    else:

        raise TypeError("Arrays u1, u2 must have the same length")

    return integrate(u1*u2, length, N) 


# take in a simulation object (where the run has already been completed) and computes the amplitude of the internal mode
#
def amplitude(sim):

    sfrac = sim.sponge_params['spongeless_frac']

    slength, sN = sfrac*sim.length, sfrac*sim.N

    x = clip_spongeless(sim.x, sfrac)

    im = internal_mode(x)

    downstairs = L2prod(im, im, slength, sN)

    u = clip_spongeless(sim.Udata[0, :, :], sfrac)

    Nt, Nx = np.shape(u)

    amplitude = np.zeros(Nt, dtype=float)

    for n in np.arange(0, Nt):

        amplitude[n] = L2prod(u[n,:], im, slength, sN)/downstairs

    return amplitude

# now do the actual curve-fitting stuff

from scipy.optimize import curve_fit
from scipy.signal import argrelmax

nsteps = int(T / (my_sim.ndump*dt))
t = np.linspace(0, T, num=1 + nsteps, endpoint=True)

# get training data
A_simulated = amplitude(my_sim)
aa = -A_simulated # I empirically found the negative amplitudes are slightly higher, so I only use
# these for getting the training data for our fit

indices_train = argrelmax(aa)
A_train = aa[indices_train]

indices_train = np.array(indices_train)
t_train = dt*my_sim.ndump*indices_train.flatten()

# fit our own decay curve
def modelcurve(t, A, B, p):
    out = A * ((1. + B*t) ** p)
    return out

bounds = [np.array([0., 0., -1.]), np.array([1.,2.,0.1])]
params_opt, params_cov = curve_fit(modelcurve, t_train, A_train, p0=[0.1, 0.01, -0.5], maxfev=int(3e4), bounds=bounds)

print('Estimated L^2 error in fit = ', np.linalg.norm(A_train-modelcurve(t_train, *params_opt)))

A_my_model = modelcurve(t, *params_opt)

print('Optimal model parameters = ', params_opt)
print('Estimated error in parameters = ', np.sqrt(np.diag(params_cov)))

picname = 'phi4pert_amplitude_fit_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'

# and plot the results
nice_multiplot([t,t,t], [A_simulated, A_my_model, -A_my_model], r"$t$", r"$a(t)$",
               curvelabels = ['Computed', 'Empirical Fit', None],
               linestyles = ['solid', 'dashed', 'dashed'], colors = ['xkcd:teal', 'xkcd:pumpkin', 'xkcd:pumpkin'],
               linewidths = [0.8, 2.2, 2.2],
               dpi = 100, show_figure = True,
               save_figure = True, picname = picname, usetex = True)

# you may also wish to compare to the Delort-Masmoudi bound with the following code...

"""
# fit Delort-Masmoudi decay curve if desired
def DM_modelcurve(t, A):
    out = A * 0.1 / np.sqrt(1. + 0.01 * t)
    return out

DM_params_opt, DM_params_cov = curve_fit(DM_modelcurve, t_train, A_train, p0=[8.], maxfev=int(3e4))
# TODO: why do we need to take such a small tail to get DM picture looking good/persisting up to T=1e4?
A_DM = DM_modelcurve(t, *DM_params_opt)

print(DM_params_opt)
print(np.sqrt(np.diag(DM_params_cov)))  # estimated errors in parameter values
"""

# but generally I find the following empirical Delort-Masmoudi decay curve is much better asymptotically:
A_DM = params_opt[0]*(params_opt[1]**params_opt[2]) * 0.1 * ((1. + 0.01*t) ** -0.5)

picname = 'phi4pert_DM_comparison_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'

nice_multiplot([t,t], [A_my_model, A_DM], r"$t$", r"$a(t)$",
               curvelabels = ['Empirical Fit', 'Empirical DM Bound'],
               linestyles = ['dashed', 'solid'], colors = ['xkcd:pumpkin', 'xkcd:slate'],
               linewidths = [2.2, 2.2], custom_ylim=[0, 0.88],
               dpi = 100, show_figure = True,
               save_figure = True, picname = picname, usetex = True)
