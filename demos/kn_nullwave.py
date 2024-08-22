import numpy as np
from scipy.fft import rfft, irfft

import joe_lab.joe as joe
import joe_lab.initial_states as joe_initial_states

length, T, N, dt = 30., 30., 2**9, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

model_kw = 'kn_null_wave'

def symbol(k):
    return -k**2

def fourier_forcing(V, k, x, nonlinear=True):
    if int(V.size - 2) == x.size:

        pass

    else:

        raise TypeError("The array V must be 2+(size of the array x).")

    N = int(V.size - 2)

    V = np.reshape(V, (N + 2,))

    NN = int(0.5 * N) + 1

    u = irfft(V[0:NN])  # only ifft first N entries of V because of storage conventions

    ux = irfft(1j * k * V[0:NN])

    v = irfft(V[NN:N+2])

    spatial_forcing = float(nonlinear) * (ux ** 2 - v ** 2)

    out = 1j * np.zeros(N + 2, dtype=float)
    out[NN:] = rfft(spatial_forcing)

    return out

my_model = joe.model(model_kw, 2, symbol, fourier_forcing, nonlinear=True)

my_initial_state = joe_initial_states.builtin_initial_state('kdv_soliton')

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.matter', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=45, usetex=False, fieldcolor='xkcd:deep magenta', fieldname='u')
