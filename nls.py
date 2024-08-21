import numpy as np

from scipy.fftpack import fft, ifft

from joe_main_lib import model, simulation, initial_state

g = 1.

def my_symbol(k):
    return -1j*k**2

def my_fourier_forcing(V,k,x,nonlinear=True):

    u = ifft(V)

    out =  fft(float(nonlinear)*1j*g*u*np.absolute(u)**2)

    return out

model_kw = 'focusing_nls'

# if the soln is complex, this must be told to the program explicitly
my_model = model(model_kw, 1, my_symbol, my_fourier_forcing, nonlinear=True, complex=True)

def plane_wave(x):
    return np.ones_like(x) + 1j*np.zeros_like(x)

def nls_soliton(x,a=1., c=1.):

    out = 1j*np.zeros_like(x)

    xmax = 180
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = np.sqrt(2.*a)*np.exp(1j*0.5*c*x)/np.cosh(np.sqrt(a)*x)

    return out

my_initial_state = initial_state('nls_soliton', nls_soliton)
#initial_state('plane_wave', plane_wave) #

length, T, N, dt = 100., 100., 2**8, 1e-2
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=10)

my_sim.load_or_run(method_kw='etdrk4', save=True)

my_sim.hov_plot(show_figure=False, save_figure=True, usetex=False, cmap='plasma')
my_sim.hov_plot_modulus(show_figure=False, save_figure=True, usetex=False, cmap='RdPu')
my_sim.save_movie(fps=100, usetex=False, fieldcolor='xkcd:heliotrope')
my_sim.save_movie_modulus(fps=100, usetex=False, fieldcolor='xkcd:barney purple')
