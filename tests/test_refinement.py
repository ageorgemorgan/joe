import numpy as np

import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

length, T, N, dt = 32.*np.pi, 150., 2**7, 2**-6
stgrid = {'length':length, 'T':T, 'N':N, 'dt':dt}
my_model = joe_models.builtin_model('ks', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('ks_chaos')

# do refinement study on ks to verify accuracy
nmin, nmax = 1, 9
Ns = np.array([2**6, 2**7, 2**8])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
slope = joe.do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic',
                                show_figure=False, save_figure=False, usetex=False)

assert 3.8 <= slope
