import numpy as np

import joe_main_lib


def initial_state(x, initial_state_kw):
    amp = 0.1
    x0 = 0.
    k0 = 1.
    width = 1.

    if initial_state_kw == 'gaussian_even':

        out = 6.*np.exp(-x**2)

    elif initial_state_kw == 'gaussian_even_alt':

        out = 1.3*np.exp(-x**2)

    elif initial_state_kw == 'kdv_soliton':

        c = 2.

        out = 0.5*c*(np.cosh(0.5*np.sqrt(c)*(x))**-2)

    elif initial_state_kw == 'gaussian_odd':

        out = amp * (np.sin(k0 * x)) * np.exp(-width * (x - x0) ** 2)

    elif initial_state_kw == 'gaussian_no_parity':

        out = amp * (0.7 * np.sin(k0 * x) + 0.3 * np.cos(x)) * np.exp(-width * (x - x0) ** 2)

    elif initial_state_kw == 'translational_mode':

        out = np.cosh(x / np.sqrt(2)) ** -2

    elif initial_state_kw == 'internal_mode':

        out = amp*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    elif initial_state_kw == 'tritone':

        a = 1.2*np.sqrt(2.)  # this value gives the Getmanov tri-tone!

        out = a*np.sinh(x / np.sqrt(2)) * (np.cosh(x / np.sqrt(2))) ** -2

    elif initial_state_kw == 'trivial':

        pass

    elif initial_state_kw == '0_energy':

        out = -1. + 3. * np.tanh(x / np.sqrt(2)) ** 2

    elif initial_state_kw == 'ks_chaos':

        out = np.cos((x+16.*np.pi)/16.) * (1. + np.sin((x+16.*np.pi) / 16.))

    elif initial_state_kw == 'bbm_weird_wavepacket':

        out = (0.1*np.cos(20.*x) + np.cos(0.2*x))*np.exp(-x**2)

    else:

        raise NameError("Invalid initial state keyword string. Acceptable keywords: gaussian_even, gaussian_odd, "
                        "gaussian_no_parity, translational_mode, internal_mode, tritone, 0_energy, trivial")

    return out


def builtin_initial_state(initial_state_kw):

    def my_initial_state_func(x):
        return initial_state(x, initial_state_kw)

    out = joe_main_lib.initial_state(initial_state_kw, my_initial_state_func)

    return out
