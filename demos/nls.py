import numpy as np

from scipy.fft import fft, ifft

import joe_lab.joe as joe

g = 1.

def my_symbol(k):
    return -1j*k**2

def my_fourier_forcing(V,k,x,nonlinear=True):

    u = ifft(V)

    out =  fft(float(nonlinear)*1j*g*u*np.absolute(u)**2)

    return out

model_kw = 'focusing_nls'

# if the soln is complex, this must be told to the program explicitly
my_model = joe.model(model_kw, 1, my_symbol, my_fourier_forcing, nonlinear=True, complex=True)

def plane_wave(x):
    return np.ones_like(x) + 1j*np.zeros_like(x)

def nls_soliton(x,a=1., c=1.):

    out = 1j*np.zeros_like(x)

    xmax = 180
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = np.sqrt(2.*a)*np.exp(1j*0.5*c*x[abs(x) <= xmax])/np.cosh(np.sqrt(a)*x[abs(x) <= xmax])

    return out

my_initial_state = joe.initial_state('nls_soliton', nls_soliton)
#initial_state('plane_wave', plane_wave) #

length, T, N, dt = 400., 400., 2**10, 1e-2
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}

l_endpt = -length * 0.5
r_endpt = l_endpt + 4e-4 * length
width = (2 ** -6) * length / 100.
sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                 'width': width, 'expdamp_freq': 2,
                 'damping_amplitude': 30.,
                 'splitting_method_kw': 'naive',
                 'spongeless_frac': .5}  # this is the fraction of the middle of the spatial domain to keep in the plots

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params, ndump=20)
#my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.load_or_run(method_kw='etdrk4', save=False)

#my_sim.hov_plot(show_figure=False, save_figure=True, usetex=False, cmap='plasma')
my_sim.hov_plot_modulus(show_figure=True, save_figure=True, usetex=False, cmap='RdPu')
#my_sim.save_movie(fps=100, usetex=False, fieldcolor='xkcd:heliotrope')
#my_sim.save_movie_modulus(fps=100, usetex=False, fieldcolor='xkcd:barney purple')
#my_sim.save_combomovie(fps=100, usetex=False, fieldcolor='xkcd:barney purple')

"""
nmin, nmax = 2, 12
Ns = np.array([2**8, 2**9, 2**10])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))

joe.do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw='etdrk4',
                    show_figure=True, save_figure=True, usetex=True, fit_min=3, fit_max=9)
"""
