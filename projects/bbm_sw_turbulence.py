import numpy as np
np.random.seed(32)

from scipy.fft import rfft, irfft, rfftfreq

import joe_lab.joe as joe
from joe_lab.visualization import nice_plot
from joe_lab.utils import integrate

def sample_one_phase(length):
    return np.random.uniform(low=-0.5 * length, high=0.5 * length)
def get_sample_phases(m, length, min_dist=20.):
    x = np.zeros(m)

    x[0] = sample_one_phase(length)

    for k in range(1, m):

        draw_new_sample = True

        while draw_new_sample:

            y = sample_one_phase(length)

            ds = np.array([np.abs(y - xx) >= min_dist for xx in x[0:k]])

            bdry_ds = np.array([np.abs(y - xx) >= 0.5*min_dist for xx in np.array([-0.5*length, 0.5*length])])

            if ds.all() and bdry_ds.all():

                draw_new_sample = False

        x[k] = y

    return x

S = 1.
delta = 2.

a_crit = -3./(delta*S) # amplitude below which we see negative solitary waves

def bbm_solitary_wave(x,c=1.):
    out = np.zeros_like(x, dtype=float)
    xmax = 180
    out[abs(x) > xmax] = 0.
    xx = x[abs(x) <= xmax]
    out[abs(x) <= xmax] = (3. / S) * c * (np.cosh(0.5 * np.sqrt(c / (1. + delta * c)) * (xx)) ** -2)
    return out

def my_symbol(k):
    return 1j * (k ** 3) / (1. + delta*k ** 2)

def my_fourier_forcing(V,k,x,nonlinear=True):
    p = 1.

    out = -S * float(nonlinear) * (1. / (p + 1.)) * (1j * k / (1. + delta*k ** 2) )*  rfft(irfft(V) ** (p + 1))

    return out

def soliton_gas_ic(x,m,length):

    out = 0.

    #phases = get_sample_phases(m, length, min_dist=0.033*length)
    phases = np.linspace(-0.5 * length + 30, 0.5 * length - 30, num=m, endpoint=True)

    amps = np.random.normal(loc=1., scale=np.sqrt(0.2), size=m) #np.random.uniform(low=.5, high=1.4, size=m)

    # mm = int(0.5*m)

    # amps_plus = np.random.uniform(low=.5, high=1.4, size=mm)

    # amps_minus = np.random.uniform(low=a_crit-0.5, high=a_crit-0.3, size=mm)

    # amps = np.concatenate((amps_plus, amps_minus))

    cs = (S/3.)*amps

    np.random.shuffle(cs)

    for k in range(0,m):

        out += bbm_solitary_wave(x-phases[k], c=cs[k])

    return out

# params are set to be at 1/4 of what DP2014 had in terms of spatial grid, so we have 1/4 of the solitons
# as well
length, T, N, dt = 2.*1395., 2., 2**13, 4e-3
m = 50 # number of solitons in the gas

ndump = int(1./dt)

stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
my_model = joe.model('bbm_dp2014', 1, my_symbol, my_fourier_forcing, nonlinear=True)
my_initial_state = joe.initial_state('soliton_gas_gaussian', lambda x: soliton_gas_ic(x, m, length))

my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=ndump)

#my_sim.plot_initial_condition(show_figure=True, save_figure=False)

my_sim.load_or_run(method_kw='etdrk4', print_runtime=True, save=False)
my_sim.hov_plot(cmap = 'cmo.haline', usetex=True, save_figure=False, show_figure=True)
#my_sim.save_movie(dpi=200, fps=200, usetex=False, fieldcolor='xkcd:cerulean', fieldname='u')


my_sim.get_fm()
fm_error = np.amax(my_sim.fm_error)
print(fm_error)


# compute the BBM energy (Sobolev H^1 norm)
def energy(u):
    # get wavenumbers for the grid of S^1 with N samples
    k = 2. * np.pi * N * rfftfreq(N) / length

    spring = irfft(1j * k * rfft(u)) ** 2

    out = integrate(u**2 + delta*spring, length, N)

    return out

# get the energies associated to each time
times = np.linspace(0., T, num=1 + int(T / (dt*my_sim.ndump)), endpoint=True)
Udata = my_sim.Udata
E = energy(Udata)
E_error = E-E[0]
E_error_rel = E_error/E[0]
print('Max absolute error in energy = ', np.amax(np.abs(E_error)))
print('Max relative error in energy =', np.amax(np.abs(E_error_rel)))

# draw the figure
dpi = 400
picname = 'bbm_dp2014_st_gaussian_energy_test_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=' + my_sim.initial_state_kw  + '.png'
nice_plot(times, E_error, r'$t$', r'Error in Energy', dpi=dpi, show_figure=False,
          save_figure=True, picname=picname, linestyle='solid', color='xkcd:blueberry', usetex=True)

picname = 'bbm_dp2014_st_gaussian_energy_test_rel_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (length, T, N, dt) + '_ICkw=' + my_sim.initial_state_kw  + '.png'
nice_plot(times, E_error, r'$t$', r'Relative Error in Energy', dpi=dpi, show_figure=False,
          save_figure=True, picname=picname, linestyle='solid', color='xkcd:blueberry', usetex=True)
