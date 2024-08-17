import numpy as np
from scipy.fftpack import fft, ifft

from joe_main_lib import simulation, model, initial_state
from models import builtin_model
from initial_states import builtin_initial_state, sinegordon_soliton

length, T, N, dt = 30.,30., 2**9, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

model_kw = 'kn_null_wave'

def symbol(k):
    return -k**2

def fourier_forcing(V, k, x, nonlinear=True):
    if int(0.5 * V.size) == x.size:

        pass

    else:

        raise TypeError("The array V must be twice the length of the array x.")

    N = int(0.5 * np.size(V))

    V = np.reshape(V, (2 * N,))

    ux = np.real(ifft(1j*k*V[0:N]))

    v = np.real(ifft(V[N:2*N]))

    spatial_forcing = float(nonlinear)*(ux**2 - v**2)

    out = 1j * np.zeros(2 * N, dtype=float)
    out[N:] = fft(spatial_forcing)

    return out

my_model = model(model_kw, 2, symbol, fourier_forcing, nonlinear=True)

my_initial_state = builtin_initial_state('kdv_soliton')

my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save_pkl=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.matter', fieldname='u', show_figure=True, save_figure=True, usetex=True)
my_sim.save_movie(dpi=200, fps=45, usetex=False, fieldcolor='xkcd:deep magenta', fieldname='u')