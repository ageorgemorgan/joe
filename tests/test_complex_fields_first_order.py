import numpy as np
from scipy.fft import fft, ifft

import joe_lab.joe as joe

from testing_utils import do_it_all

g = 1.

def my_symbol(k):
    return -1j*k**2

def my_fourier_forcing(V,k,x,nonlinear=True):

    u = ifft(V)

    out =  fft(float(nonlinear)*1j*g*u*np.absolute(u)**2)

    return out

model_kw = 'focusing_nls'
my_model = joe.model(model_kw, 1, my_symbol, my_fourier_forcing, nonlinear=True, complex=True)

def nls_soliton(x,a=1., c=1.):

    out = 1j*np.zeros_like(x)

    xmax = 180
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = np.sqrt(2.*a)*np.exp(1j*0.5*c*x[abs(x) <= xmax])/np.cosh(np.sqrt(a)*x[abs(x) <= xmax])

    return out

my_initial_state = joe.initial_state('nls_soliton', nls_soliton)

length, T, N, dt = 400., 8., 2 ** 10, 1e-2
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

def test_periodic_bcs():
    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

    converged = do_it_all(my_sim)

    # test that the code actually ran
    assert converged == True

def test_sponge_layer_bcs():
    l_endpt = -length * 0.5
    r_endpt = l_endpt + 4e-4 * length
    width = (2 ** -6) * length / 100.
    sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                     'width': width, 'expdamp_freq': 2,
                     'damping_amplitude': 30.,
                     'splitting_method_kw': 'naive',
                     'spongeless_frac': .5}

    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params,
                                ndump=20)
    converged = do_it_all(my_sim)

    # test that the code actually ran
    assert converged == True