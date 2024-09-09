# joe code for reproducing the experiments and (most) statistical results from a paper of Didenkulova
# https://www.sciencedirect.com/science/article/abs/pii/S0167278918305657 BUT
# using the non-integrable Gardner-BBM instead of the "regular" Gardner eqn

import time

import numpy as np

from scipy.signal import argrelmin, argrelmax # for postprocessing: amplitude hist and cdf
from scipy.stats import ecdf # again for postprocessing

import joe_lab.joe as joe
from joe_lab.utils import integrate, dealiased_pow
from joe_lab.visualization import spinner, nice_plot, nice_multiplot, nice_hist

from joblib import Parallel, delayed

##############################################################################################
########################## STAGE 1: SIMULATION OF THE ENSEMBLE ###############################
##############################################################################################

# fix basic params
num_samples = 1
num_waves = 30

ndump = 500

# get stgrid
length, T, N, dt = 930., 200., 2 ** 13, 2e-5
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

# get model
def my_symbol(k):
    return 1j * (k ** 3) / (1. + k ** 2)

def my_fourier_forcing(V,k,x,nonlinear=True):

    Fu2 = dealiased_pow(V,2)
    Fu3 = dealiased_pow(V,3)

    out = 6. * float(nonlinear) * ((1j * k) / (1. + k ** 2)) * (
            0.5 * Fu2 - (1. / 3.) * Fu3)

    return out

my_model = joe.model('gardner-bbm', 1, my_symbol, my_fourier_forcing, nonlinear=True)

# get initial state
def gardnerbbm_solitary_wave(x, c = 1., p = 1.):

    out = np.zeros_like(x, dtype=float)
    xmax = 7e2
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c/(1.+c)) * x[abs(x) <= xmax]))

    return out

def sw_gas_ic(x, m):
    out = 0.

    phases = np.linspace(-0.5 * length + 15, 0.5 * length - 15, num=m, endpoint=True)

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
        out += gardnerbbm_solitary_wave(x - phases[k], c=z[k, 0], p=z[k, 1])

    return out

# define a function that computes the H1 energy
from scipy.fft import rfft, irfft, rfftfreq

def energy(u):
    # get wavenumbers for the grid of S^1 with N samples
    k = 2. * np.pi * N * rfftfreq(N) / length

    spring = irfft(1j * k * rfft(u)) ** 2

    out = 0.5*integrate(np.absolute(u)**2 + spring, length)

    return out

import matplotlib.pyplot as plt
num_timesteps = 1+int(T/(dt*ndump))
tt = ndump*dt*np.arange(0, num_timesteps)

# define a function that takes in a sample number and does a sim
def sample_st(sample):
    ic_string = 'st_sample_' + str(sample)

    np.random.seed(32)  # fix seed for reproducibility if debugging/checking time step
    my_initial_state = joe.initial_state(ic_string, lambda x: sw_gas_ic(x, num_waves))

    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=ndump)
    my_sim.plot_initial_condition(show_figure=True, save_figure=False)
    my_sim.load_or_run(print_runtime=False, verbose=False, save=True)

    # error in mass
    my_sim.get_fm()
    fm_error = np.amax(my_sim.fm_error)

    # error in energy
    E = energy(my_sim.Udata)
    print(E[0])
    #plt.plot(tt, E-E[0]) # (E-E[0])/E[0])
    #plt.show()
    E_error = np.amax(np.abs(E-E[0])/E[0])
    return np.array([fm_error, E_error])

start = time.time()

with spinner('Simulating Gardner-BBM solitary wave turbulence...'):
    errors = Parallel(n_jobs=1)(delayed(sample_st)(sample) for sample in range(1, num_samples+1))
    #(25+1, num_samples+1) (1, num_samples+1)
    # depending on your machine, n_jobs or the number of batches you split your ensemble into may
    # need to be altered!
    errors = np.array(errors)
    fm_errors = errors[:, 0]
    E_errors = errors[:, 1]

end = time.time()
runtime = end - start
print('Runtime for Gardner soliton turbulence simulation = %.4f' % runtime + ' s')

print('Maximum error in first moment over ensemble = ', np.amax(fm_errors))
print('Maximum error in energy over ensemble = ', np.amax(E_errors))

##############################################################################################
########################## STAGE 2: PREP FOR POSTPROCESSING (HELPER FNCS ETC.) ###############
##############################################################################################

num_timesteps = 1+int(T/(dt*ndump))
tt = ndump*dt*np.arange(0, num_timesteps)

# code for computing higher moments given field and its first two moments
def get_higher_moments(u,fm,sm):

    v = (u-fm)**3

    skew = (1./length)*integrate(v, length)
    skew /= sm**1.5

    v = (u-fm)**4

    kurt = (1./length)*integrate(v, length)
    kurt /= sm**2

    return skew, kurt

# code for computing CDF of unif distribution on [a,b]
def unif_cdf(x, a=-1, b=1.):
    out = np.zeros_like(x, dtype=float)
    out[x>a] = (x[x>a] - a)/(b-a)
    out[x>=b] = 1.
    return out

##############################################################################################
########################## STAGE 3: POSTPROCESSING ###########################################
##############################################################################################

# init storage
skew_store = np.zeros((num_samples, num_timesteps), dtype=float)
kurt_store = np.zeros((num_samples, num_timesteps), dtype=float)
max_storage = np.zeros((num_samples, num_timesteps), dtype=float)
min_storage = np.zeros((num_samples, num_timesteps), dtype=float)
pos_amp = []
neg_amp = []

for sample in range(1, num_samples+1):

    ic_string = 'st_sample_' + str(sample)
    my_initial_state = joe.initial_state(ic_string, lambda x: sw_gas_ic(x, num_waves))
    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=ndump)
    my_sim.load()

    #"""
    # if desired, save an example initial state plot and Hovmoeller plot for one sample
    if sample == 1:
        my_sim.plot_initial_condition(color='xkcd:cerulean', usetex=True, show_figure=False, save_figure=True)

        #my_sim.hov_plot(umin=-3., umax=3., dpi=600, usetex=True, save_figure=True, show_figure=False, cmap='cmo.thermal')
        # after a lot of experimenting I really think the thermal colormap is the right way to go
        # for Gardner, where the really thin antisolitons need to stand out as strongly as possible

        my_sim.save_combomovie(dpi=80, fps=400, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

        #E = energy(my_sim.Udata)
        
        #picname = 'gardnerbbm_st_energy_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
        #nice_plot(tt, (E-E[0])/E[0], r"$t$", r"Relative Error in Energy", custom_ylim=None,
        #  dpi=600,show_figure=False, save_figure=True, picname=picname, linestyle='solid', color='xkcd:reddish purple',
        #  usetex=True)
    #"""

    u = my_sim.Udata

    # moments
    my_sim.get_fm()
    fm = my_sim.fm[0]

    my_sim.get_sm()
    sm = my_sim.sm

    skew, kurt = get_higher_moments(my_sim.Udata, fm, sm)
    skew_store[sample-1, :] = skew
    kurt_store[sample-1, :] = kurt

    # +/- amplitude asymmetry plot
    umax = np.amax(u, axis=1)
    umin = np.amin(u, axis=1)

    max_storage[sample-1] = umax
    min_storage[sample-1] = umin

    # amplitude histograms and CDF

    # need to set a "mixing time" after which we assume the "gas" is well-mixed and we're OK to start doing some data
    # analysis. Honestly 0.25*T could even be a bit large!
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

import sys
sys.exit()

##############################################################################################
########################## STAGE 4: PRODUCE POSTPROCESSING PLOTS #############################
##############################################################################################

dpi = 600

# Plot skew and kurtosis averaged over the ensemble
picname = 'gardnerbbm_st_skew_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_plot(tt, np.mean(skew_store, axis=0), r"$t$", r"Skewness", custom_ylim=None,
          dpi=dpi,show_figure=False, save_figure=True, picname=picname, linestyle='solid', color='xkcd:pumpkin',
          usetex=True)

picname = 'gardnerbbm_st_kurt_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_plot(tt, np.mean(kurt_store, axis=0), r"$t$", r"Kurtosis", custom_ylim=None, dpi=dpi,
          show_figure=False, save_figure=True, picname=picname, linestyle='solid',
          color='xkcd:pinky purple', usetex=True)

# Plot +/- amplitudes (max/min over ensemble resp) to illustrate asymmetry between polarities
picname = 'gardnerbbm_st_Amps_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_multiplot([tt,tt], [np.max(max_storage, axis=0), np.min(min_storage, axis=0)],
               r"$t$", r"Min/Max Amplitudes",
               curvelabels = ['Max', 'Min'],
               linestyles = ['dotted', 'dashed'], colors = ['xkcd:deep pink', 'xkcd:teal green'], linewidths = [2, 2],
               custom_ylim=[-6,4], dpi = dpi, show_figure = False, save_figure = True, picname = picname, usetex=True)

# Plot histograms of amplitudes

# + amplitudes
picname = 'gardnerbbm_st_hist+_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_hist(pos_amp, r'$A_{+}$', dpi=dpi, show_figure=False, save_figure=True, picname=picname,
              color='xkcd:deep pink', usetex=True)

# - amplitudes
picname = 'gardnerbbm_st_hist-_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_hist(neg_amp, r'$A_{-}$', dpi=dpi, show_figure=False, save_figure=True, picname=picname,
              color='xkcd:teal green', usetex=True)

# Plot amplitude CDFs

# + amplitudes
pos_cdf = ecdf(pos_amp)
xx = np.linspace(2,4, num=600)

picname = 'gardnerbbm_st_cdf+_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_multiplot([xx,xx], [pos_cdf.cdf.evaluate(xx), unif_cdf(xx, a=2.3, b=3)],
               r"$A_{+}$", r"$F\left(A_{+}\right)$",
               curvelabels = ['Observed', 'Uniform'],
               linestyles = ['solid', 'dashed'], colors = ['xkcd:deep pink', 'xkcd:blueberry'],
               linewidths = [3, 2], custom_ylim=[-0.02,1.02],
               dpi = dpi, show_figure = False, save_figure = True, picname = picname, usetex=True)

# - amplitudes
neg_cdf = ecdf(neg_amp)
xx = np.linspace(-6,0, num=600)

picname = 'gardnerbbm_st_cdf-_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '.png'
nice_multiplot([xx,xx], [neg_cdf.cdf.evaluate(xx), unif_cdf(xx, a=-3., b=-0.1)],
               r"$A_{-}$", r"$F\left(A_{-}\right)$",
               curvelabels = ['Observed', 'Uniform'],
               linestyles = ['solid', 'dashed'], colors = ['xkcd:teal green', 'xkcd:blueberry'],
               linewidths = [3, 2], custom_ylim=[-0.02,1.02],
               dpi = dpi, show_figure = False, save_figure = True, picname = picname, usetex=True)
