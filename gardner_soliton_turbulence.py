import numpy as np
import time

from joe_main_lib import simulation, initial_state
from models import builtin_model
from visualization import spinner

from joblib import Parallel, delayed

#np.random.seed(32)

# fix basic params
num_samples = 50
num_waves = 30

ndump = 500

# get stgrid
length, T, N, dt = 600., 50., 2 ** 13, 2e-5
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

# get model
my_model = builtin_model('gardner', nonlinear=True)


# get initial state
# note that, for large domains, cosh(x near endpts) may encounter overflow, so it's good to just manually set
# the tails of the wave to zero to avoid these concerns (see also the great discussion here
# https://stackoverflow.com/questions/31889801/overflow-in-numpy-cosh-function
# which has inspired my approach below)
def gardner_soliton(x, c=1., p=1.):
    out = np.zeros_like(x, dtype=float)
    xmax = 180
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c) * x[abs(x) <= xmax]))
    return out


def soliton_gas_ic(x, m):
    out = 0.

    phases = np.linspace(-0.5 * length + 10, 0.5 * length - 10, num=m, endpoint=True)

    mm = int(0.5 * m)

    amps_plus = np.random.uniform(low=2.3, high=3., size=mm)

    cs_plus = (amps_plus - 1.) ** 2 - 1.  # using the Gardner NL dispersion relation

    amps_minus = -1. * np.random.uniform(low=.1, high=3., size=mm)

    cs_minus = (np.abs(amps_minus) + 1.) ** 2 - 1.  # using the Gardner NL dispersion relation

    cs = np.concatenate((cs_plus, cs_minus))

    ps = np.ones(m, dtype=float)

    ps[mm:] *= -1

    z = np.zeros((m, 2), dtype=float)

    z[:, 0] = cs

    z[:, 1] = ps

    np.random.shuffle(z)

    for k in range(0, m):
        out += gardner_soliton(x - phases[k], c=z[k, 0], p=z[k, 1])

    return out

# define a function that takes in a sample number and does a sim
def sample_st(sample):
    ic_string = 'st_sample_' + str(sample)

    my_initial_state = initial_state(ic_string, lambda x: soliton_gas_ic(x, num_waves))

    my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=ndump)
    my_sim.load_or_run(print_runtime=False, verbose=False, save_npy=False, save_pkl=True)

    #my_sim.hov_plot(umin=-3., umax=3., dpi=600, usetex=True, save_figure=True, show_figure=False, cmap='cmo.thermal')
    # after a lot of experimenting I really think the thermal colormap is the right way to go
    # for Gardner, where the really thin antisolitons need to stand out as strongly as possible

    my_sim.get_fm()
    my_sim.get_sm()
    fm_error = np.amax(my_sim.fm_error)
    sm_error = np.amax(my_sim.sm_error)
    return np.array([fm_error, sm_error])


# initialize moment errors
fm_errors = np.zeros(num_samples, dtype=float)
sm_errors = np.zeros(num_samples, dtype=float)

start = time.time()

with spinner('Simulating Gardner soliton turbulence...'):
    #"""
    errors = Parallel(n_jobs=-2)(delayed(sample_st)(sample) for sample in range(0, num_samples))
    errors = np.array(errors)
    fm_errors = errors[:, 0]
    sm_errors = errors[:, 1]
    #"""

    """
    for sample in range(0, num_samples):
        ic_string = 'st_sample_' + str(sample)

        my_initial_state = initial_state(ic_string, lambda x: soliton_gas_ic(x, num_waves))

        my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=400)
        my_sim.load_or_run(print_runtime=False, verbose=False, save=True)

        # my_sim.hov_plot(umin=-3., umax=3., dpi=600, usetex=True, save_figure=True, show_figure=False, cmap='cmo.thermal')
        # after a lot of experimenting I really think the thermal colormap is the right way to go
        # for Gardner, where the really thin antisolitons need to stand out as strongly as possible

        my_sim.get_fm()
        my_sim.get_sm()
        fm_error = np.amax(my_sim.fm_error)
        sm_error = np.amax(my_sim.sm_error)
        fm_errors[sample] = fm_error
        sm_errors[sample] = sm_error
    """

end = time.time()
runtime = end - start
print('Runtime for Gardner soliton turbulence simulation = %.4f' % runtime + ' s')


print(np.amax(fm_errors))
print(np.amax(sm_errors))

num_timesteps = 1+int(T/(dt*ndump))
tt = ndump*dt*np.arange(0, num_timesteps)

import matplotlib.pyplot as plt
#plt.plot(range(0, np.size(sm_errors)), sm_errors)
#plt.show()

from numpy.fft import fft

#"""
# code for computing higher moments of the ensemble

def get_higher_moments(u,fm,sm):

    #fmfm = np.tile(fm, [N, 1]).T

    v = (u-fm)**3

    skew = (1. / N) * np.real(fft(v, axis=1)[:, 0])
    skew /= sm**1.5

    v = (u-fm)**4

    kurt = (1. / N) * np.real(fft(v, axis=1)[:, 0])
    kurt /= sm**2

    return skew, kurt

from scipy.signal import argrelmin, argrelmax # for amplitude hist and cdf

skew_store = np.zeros((num_samples, num_timesteps), dtype=float)
kurt_store = np.zeros((num_samples, num_timesteps), dtype=float)
max_storage = np.zeros((num_samples, num_timesteps), dtype=float)
min_storage = np.zeros((num_samples, num_timesteps), dtype=float)
pos_amp = []
neg_amp = []

for sample in range(0, num_samples):

    ic_string = 'st_sample_' + str(sample)
    my_initial_state = initial_state(ic_string, lambda x: soliton_gas_ic(x, num_waves))
    my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=ndump)
    my_sim.load()

    u = my_sim.Udata

    # skew and kurtosis
    my_sim.get_fm()
    fm = my_sim.fm[0] #TODO: better to put in "actual" or imperfect first/second moments into higher moment computations? Barely matters
    #print(fm)
    my_sim.get_sm()
    sm = my_sim.sm[0]
    #print(sm)

    skew, kurt = get_higher_moments(my_sim.Udata, fm, sm)
    skew_store[sample, :] = skew
    kurt_store[sample, :] = kurt

    # +/- amplitude asymmetry plot
    umax = np.amax(u, axis=1)
    umin = np.amin(u, axis=1)

    max_storage[sample] = umax
    min_storage[sample] = umin

    # amplitude histograms and CDF
    mixing_time = 0.25 * T

    mixing_index = int(mixing_time / (ndump * dt))

    # only count stuff after mixing time
    u = u[mixing_index:, :]

    delta_pos = 2e0  # threshold: ignore + waves below this amplitude
    delta_min = -1e-1  # same but for - waves

    pos_part = u[u >= delta_pos]
    max = argrelmax(pos_part, axis=0)
    pos_amp.extend(pos_part[max])

    neg_part = u[u <= delta_min]
    min = argrelmin(neg_part, axis=0)
    neg_amp.extend(neg_part[min])

###############################################################
############# ALL THE PLOTTING STUFF ##########################
###############################################################
import os

# add the folder "visuals" to our path... more on this below
my_path = os.path.join("visuals")

# first, if the folder doesn't exist, make it
if not os.path.isdir(my_path):
    os.makedirs(my_path)

plt.rcParams["font.family"] = "serif"
dpi = 600

try:
    plt.rc('text', usetex=True)
    usetex = True

except RuntimeError:  # catch a user error thinking they have tex when they don't
    usetex = False

# plot skew and kurtosis averaged over the ensemble
fig = plt.figure()
plt.plot(tt, np.mean(skew_store, axis=0), linewidth=2, color='xkcd:pumpkin')
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

if usetex:
    plt.xlabel(r"$t$", fontsize=22, color='k')
else:
    plt.xlabel(r"t", fontsize=22, color='k')

plt.ylabel(r"Skewness", fontsize=22, color='k')
plt.xlim([0,T])
plt.tight_layout()
picname = 'gardner_st_skew_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()

fig = plt.figure()
plt.plot(tt, np.mean(kurt_store, axis=0), linewidth=2, color='xkcd:pinky purple')
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

if usetex:
    plt.xlabel(r"$t$", fontsize=22, color='k')
else:
    plt.xlabel(r"t", fontsize=22, color='k')

plt.ylabel(r"Kurtosis", fontsize=22, color='k')
plt.xlim([0, T])
plt.tight_layout()
picname = 'gardner_st_kurt_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()

# Plot +/- amplitudes (max/min over ensemble resp) to illustrate asymmetry between polarities
fig = plt.figure()
plt.plot(tt, np.max(max_storage, axis=0), label='Max', color='xkcd:deep pink', linewidth=2, linestyle='dotted')
plt.plot(tt, np.min(min_storage, axis=0), label='Min', color='xkcd:teal green', linewidth=2)
# note: using mean over samples rather than max also gives meaningful information, perhaps worth including.
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

if usetex:
    plt.xlabel(r"$t$", fontsize=22, color='k')
else:
    plt.xlabel(r"t", fontsize=22, color='k')

plt.ylabel(r"Min/Max Amplitudes", fontsize=22, color='k')
plt.legend(fontsize=22, loc='best')
plt.xlim([0, T])
plt.ylim([-6, 4])
plt.tight_layout()
picname = 'gardner_st_Amps_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()


# plot amplitude histograms
fig = plt.figure()
plt.hist(pos_amp, color='xkcd:deep pink')
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

if usetex:
    plt.xlabel(r"$A_{+}$", fontsize=22, color='k')
else:
    plt.xlabel(r"A_{+}", fontsize=22, color='k')

plt.ylabel(r"Number of Instances", fontsize=22, color='k')
plt.xlim([2,4])
plt.tight_layout()
picname = 'gardner_st_hist+_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()

fig = plt.figure()
plt.hist(neg_amp, color='xkcd:teal green')
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')

if usetex:
    plt.xlabel(r"$A_{-}$", fontsize=22, color='k')
else:
    plt.xlabel(r"A_{-}", fontsize=22, color='k')

plt.ylabel(r"Number of Instances", fontsize=22, color='k')
plt.xlim([-6,-0.1])
plt.tight_layout()
picname = 'gardner_st_hist-_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()

# Plot amplitude CDFs
def unif_cdf(x, a=-1, b=1.):
    out = np.zeros_like(x, dtype=float)
    out[x>a] = (x[x>a] - a)/(b-a)
    out[x>=b] = 1.
    return out

from scipy.stats import ecdf

pos_cdf = ecdf(pos_amp)
fig, ax = plt.subplots()
pos_cdf.cdf.plot(ax)
xx = np.linspace(2,4, num=600)
plt.plot(xx, pos_cdf.cdf.evaluate(xx), label='Observed', color='xkcd:deep pink', linewidth=3)
ax.plot(xx, unif_cdf(xx, a=2.3, b=3), label='Uniform', linestyle='dashed', color='xkcd:blueberry', linewidth=2)
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')
plt.legend(fontsize=22, loc='upper left')
if usetex:
    plt.xlabel(r"$A_{+}$", fontsize=22, color='k')
    plt.ylabel(r"$F\left(A_{+}\right)$", fontsize=22, color='k')
else:
    plt.xlabel(r"A_{+}", fontsize=22, color='k')
    plt.ylabel(r"F(A_{+})", fontsize=22, color='k')
plt.xlim([2,4])
plt.ylim([-0.02,1.02])
plt.tight_layout()
picname = 'gardner_st_cdf+_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()


neg_cdf = ecdf(neg_amp)
fig, ax = plt.subplots()
xx = np.linspace(-6,0, num=600)
plt.plot(xx, neg_cdf.cdf.evaluate(xx), label='Observed', color='xkcd:teal green', linewidth=3)
ax.plot(xx, unif_cdf(xx, a=-3., b=-0.1), label='Uniform', linestyle='dashed', color='xkcd:blueberry', linewidth=2)
fig.set_size_inches(8, 6)
plt.tick_params(axis='x', which='both', top=False, color='k')
plt.xticks(fontsize=20, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right=False, color='k')
plt.yticks(fontsize=20, rotation=0, color='k')
plt.legend(fontsize=22, loc='upper left')
if usetex:
    plt.xlabel(r"$A_{-}$", fontsize=22, color='k')
    plt.ylabel(r"$F\left(A_{-}\right)$", fontsize=22, color='k')
else:
    plt.xlabel(r"A_{-}", fontsize=22, color='k')
    plt.ylabel(r"F(A_{-})", fontsize=22, color='k')
plt.xlim([-6,-0.1])
plt.ylim([-0.02,1.02])
plt.tight_layout()
picname = 'gardner_st_cdf-_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=dpi)
plt.clf()

"""
uu = u[-1,:]
plt.plot(my_sim.x, uu)

    
pos_part = uu[u[-1,:]>delta] # threshold to avoid false local maxes/exclude very small waves
max = argrelmax(pos_part) # local maximizers (up to threshold)
amp = pos_part[max] # actual + wave amplitudes

# draw for validation
x_plt = []
for a in amp:
    x_plt.append(np.where(uu == a))
x_plt = -0.5*length+(length/N)*np.array(x_plt)

plt.scatter(x_plt, amp, marker='o', color='xkcd:deep pink')

neg_part = uu[u[-1, :] < -delta]
min = argrelmin(neg_part) # local minimizers (up to threshold)
amp = neg_part[min]  # actual - wave amplitudes

# draw for validation
x_plt = []
for a in amp:
    x_plt.append(np.where(uu == a))
x_plt = -0.5*length+(length/N)*np.array(x_plt)

plt.scatter(x_plt, amp, marker='v', color='xkcd:teal green')
    
plt.show()
"""