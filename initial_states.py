import numpy as np

from .joe import initial_state


def kdv_soliton(x, c=1.):
    if c <= 0:
        raise ValueError('KdV soliton speed must be strictly positive.')
    return 0.5*c*(np.cosh(0.5 * np.sqrt(c) * x) ** -2)


def gardner_soliton(x, c=1., p=1.):
    if c <= 0:
        raise ValueError('Focusing Gardner soliton speed must be strictly positive.')
    out = np.zeros_like(x, dtype=float)
    xmax = 180
    out[abs(x) > xmax] = 0.
    out[abs(x) <= xmax] = c / (-1. + p * np.sqrt(1. + c) * np.cosh(np.sqrt(c) * x[abs(x) <= xmax]))
    return out


def bbm_solitary_wave(x, c=1.):
    if c <= 0 and c >= -1:
        raise ValueError('BBM soliton speed must be strictly positive, or strictly less than -1.')
    return 0.5*c*(np.cosh(0.5 * np.sqrt(c/(1.+c)) * x) ** -2)

def sinegordon_soliton(x, c=0., p=1.):
    if np.abs(c) >= 1:
        raise ValueError('Sine-Gordon soliton speed must be in the interval (-1,1).')
    out = np.zeros_like(x, dtype=float)
    arg = p*x/np.sqrt(1.-c**2)
    xmax = 1.8e2
    out[arg > xmax] = 2.*np.pi
    out[arg < -xmax] = 0.
    out[np.abs(arg)<= xmax] = 4.*np.arctan(np.exp(arg[np.abs(arg)<= xmax]))
    return out

def sinegordon_soliton_speed(x, c=0., p=1.): # speed is at t=0
    if np.abs(c) >= 1:
        raise ValueError('Sine-Gordon soliton speed must be in the interval (-1,1).')
    out = np.zeros_like(x, dtype=float)
    gamma = 1./np.sqrt(1.-c**2)
    arg = p*x*gamma
    xmax = 1.8e2
    out[arg > xmax] = 0.
    out[arg < -xmax] = 0.
    aa = arg[np.abs(arg)<= xmax]
    out[np.abs(arg)<= xmax] = -(c*4.*p*gamma)*(np.exp(aa)/(1.+np.exp(2.*aa)))
    return out


def initial_state_func(x, initial_state_kw):
    amp = 0.1
    x0 = 0.
    k0 = 1.
    width = 1.

    if initial_state_kw == 'sine':
        k = 1.5
        out = 1.0*np.sin(k*x)

    elif initial_state_kw == 'gaussian_even':
        out = 6.*np.exp(-x**2)

    elif initial_state_kw == 'gaussian_even_alt':
        out = 1.3*np.exp(-x**2)

    elif initial_state_kw == 'kdv_soliton':
        c = 2.
        out = kdv_soliton(x, c=c)

    elif initial_state_kw == 'kdv_multisoliton':
        c0 = 3.2
        c1 = 2.5
        c2 = 1.
        out = kdv_soliton(x+80., c=c0) + kdv_soliton(x+50., c=c1) + kdv_soliton(x+10, c=c2)

    elif initial_state_kw == 'gardner_soliton':
        c = 3.
        p = 1.
        out = gardner_soliton(x, c=c, p=p)

    elif initial_state_kw == 'bbm_solitary_wave':
        c = 2.
        out = bbm_solitary_wave(x,c=c)

    elif initial_state_kw == 'bbm_multisolitary':
        c0 = 2.5
        c1 = 2.
        out = bbm_solitary_wave(x+80., c=c0) + bbm_solitary_wave(x+50., c=c1)

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

    elif initial_state_kw == 'sinegordon_soliton_interaction':
        out = np.zeros((2,np.shape(x)[0]), dtype=float)
        out[0,:] = sinegordon_soliton(x+20, c=0.9, p=1) - sinegordon_soliton(x-20, c=-0.9, p=1)
        out[1,:] = sinegordon_soliton_speed(x+20, c=0.9, p=1) - sinegordon_soliton_speed(x-20, c=-0.9, p=1)

    elif initial_state_kw == 'sinegordon_soliton_interaction_alt':
        out = np.zeros((2,np.shape(x)[0]), dtype=float)
        out[0,:] = sinegordon_soliton(x+20, c=0.9, p=1) - sinegordon_soliton(x-20, c=0., p=1)
        out[1,:] = sinegordon_soliton_speed(x+20, c=0.9, p=1)

    else:
        raise NameError("User-defined initial state keyword string not among built-in options.")

    return out

def builtin_initial_state(initial_state_kw):

    return initial_state(initial_state_kw, lambda x : initial_state_func(x, initial_state_kw))
    
