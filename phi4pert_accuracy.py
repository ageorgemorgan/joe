import numpy as np
from numpy.fft import fft, ifft, fftfreq

from joe_main_lib import integrate, simulation, do_refinement_study
from models import K0, V0, builtin_model
from initial_states import builtin_initial_state
from visualization import nice_plot

# first prescribe all the simulation parameters etc.
length, T, N, dt = 240.,100., 2**9, 1e-2

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = builtin_model('phi4pert', nonlinear=True)
my_initial_state = builtin_initial_state('gaussian_odd')

# obtain the relevant simulation, either by loading it up again or by running it.
my_sim = simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)
my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=True)

# produce plots and movies
my_sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=True, save_figure=True, usetex=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')

# Now we do all the post-processing energy analysis, and an accuracy test

# Define a function that computes the energy of our state [u,ut] at time t
x = my_sim.x
Udata = my_sim.Udata

def energy(u, ut):
    kin = ut ** 2

    # get wavenumbers for the grid of S^1 with N samples
    k = 2. * np.pi * N * fftfreq(N) / length

    spring = np.real(ifft(1j * k * fft(u))) ** 2

    potential = (2. + V0(x)) * (u ** 2) + 1.*(2. * K0(x) * u ** 3 + 0.5 * u ** 4)

    e_dens = kin + spring + potential

    out = 0.5*integrate(e_dens, length, N)

    return out

# get the energies associated to each time
times = np.linspace(0., T, num=1 + int(T / (dt*my_sim.ndump)), endpoint=True)
E = energy(Udata[0, :, :],  Udata[1, :, :])

# draw the figure
dpi = 400
picname = 'phi4pert_energy_test_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=' + my_sim.initial_state_kw  + '.png'
nice_plot(times, (E-E[0])/E[0], r'$t$', r'Relative Error in Energy', dpi=dpi, show_figure=True,
          save_figure=True, picname=picname, linestyle='solid', color='xkcd:blueberry', usetex=True)

# perform Richardson accuracy test
nmin, nmax = 4, 9
Ns = np.array([2**7, 2**8, 2**9, 2**10])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))

do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw='etdrk4',
                    show_figure=False, save_figure=True, usetex=True, dpi=dpi, fit_min=0, fit_max=4)