# Code for accuracy assessment, testing how well joe can track a KdV soliton on a periodic grid
# for a long time.

import numpy as np

from joe_main_lib import do_refinement_study
from models import builtin_model
from initial_states import builtin_initial_state

length, T = 100., 180.
my_model = builtin_model('kdv', nonlinear=True)
my_initial_state = builtin_initial_state('kdv_soliton')
method_kw='etdrk4'

nmin, nmax = 2, 14
Ns = np.array([2**7, 2**8])
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))

do_refinement_study(my_model, my_initial_state, length, T, Ns, dts, bc='periodic', method_kw=method_kw,
                    show_figure=True, save_figure=True, usetex=True, fit_min=3, fit_max=9)