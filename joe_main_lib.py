import pickle
import os

import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

from time_stepper import do_time_stepping
from initial_states import initial_state
from visualization import hov_plot, save_movie, save_combomovie, spinner, plot_refinement_study, nice_plot
from sponge_layer import clip_spongeless

# helper function for integration (of real part of fnc) in space. Uses FFT to accurately integrate over spatial domain:
# accuracy vastly beats trapezoidal rule.
# u = array storing node values of field to be integrated (last dimension of the array is spatial)
# length = length of domain
# N = number of samples of u we take (= number of grid pts)
def integrate(u, length, N):
    return (length/N) * np.real(fft(u, axis=-1)[..., 0])

# a class for models. Note how the init takes in two callables for the symbol and forcing terms: to avoid making the
# weird no-no of having callable attributes, we use a trick from
# https://stackoverflow.com/questions/35321744/python-function-as-class-attribute-becomes-a-bound-method
# and instead make the "callable attributes" !dicts! with callable entries. This dirty trick is all under the hood
# in the model class, so when defining and using a model object the user doesn't need to care.
class model:
    def __init__(self, model_kw, t_ord, symbol, fourier_forcing, nonlinear=True, complex=False):
        self.model_kw = model_kw
        self.t_ord = t_ord  # an integer
        self.symbol = {'symbol': symbol}  # callable
        self.fourier_forcing = {'fourier_forcing': fourier_forcing}  # callable
        self.nonlinear = nonlinear
        self.complex = complex

        # this defines a model PDE with the name 'model_kw' in the Fourier-space form

        # (d_t)^{t_ord} V + symbol*V = fourier_forcing

        # where V is the Fourier state.

    # obtain the actual symbol on a given mesh
    def get_symbol(self, *args):
        return self.symbol['symbol'](*args)

    # obtain the forcing term in Fourier space
    def get_fourier_forcing(self, *args):
        return self.fourier_forcing['fourier_forcing'](*args)


# turn initial states into object as well, following the syntax used to define the model class. This is really there
# instead of kw's only just to allow the user to easy input custom initial states on the fly.
class initial_state:
    def __init__(self, initial_state_kw, initial_state_func):
        self.initial_state_kw = initial_state_kw
        self.initial_state_func = {'initial_state_func': initial_state_func}
        self.initial_state = None

    # obtain the actual initial state fnc
    def get_initial_state(self, *args):
        return self.initial_state_func['initial_state_func'](*args)


# a class for simulations. You init with a "stgrid" (space-time grid) dict, a "model" object to specify the PDE,
# and an "initial_state" object. Then you can call a run simulation function on a simulation object to solve
# the IVP and store/save the soln, or load up the soln from an experiment you've already done.

class simulation:
    def __init__(self, stgrid, model, initial_state, bc, sponge_params=None, ndump=10):
        self.length = stgrid['length']
        self.T = stgrid['T']
        self.N = stgrid['N']
        self.dt = stgrid['dt']
        self.model = model  # a model object
        self.model_kw = model.model_kw
        self.t_ord = model.t_ord  # an integer
        self.initial_state_kw = initial_state.initial_state_kw
        self.nonlinear = model.nonlinear
        self.sponge_params = sponge_params
        self.complex = model.complex

        if bc == 'sponge_layer':
            self.sponge_layer = True
        elif bc == 'periodic':
            self.sponge_layer = False
        else:
            raise ValueError('User-defined BC string not accepted. Valid BC strings: periodic, sponge_layer')

        # the "spongeless fraction" attribute is a bit special for plotting and so deserves to be
        # singled out early on
        try:
            self.sfrac = self.sponge_params['spongeless_frac']
        except TypeError:
            self.sfrac = 1.


        self.ndump = ndump  # hyperparameter describing how often we save our time steps
        self.x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.N, endpoint=False)
        self.initial_state = initial_state.get_initial_state(self.x) # IMPORTANT: self.initial_state is an array, but the
        # actual input initial_state to the simulation class is an initial_state object!

        my_string = '_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                       self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_sponge_layer=' + str(self.sponge_layer)

        self.filename = 'simdata' + my_string + '.pkl'
        self.ic_picname = 'ic_plot' + my_string + '.png'
        self.picname = 'hovplot' + my_string + '.png'
        self.realpicname = 'hovplot_real' + my_string + '.png'
        self.imagpicname = 'hovplot_imag' + my_string + '.png'
        self.modpicname = 'hovplot_mod' + my_string + '.png'
        self.moviename = 'movie' + my_string + '.mp4'
        self.realmoviename = 'movie_real' + my_string + '.mp4'
        self.imagmoviename = 'movie_imag' + my_string + '.mp4'
        self.modmoviename = 'movie_mod' + my_string + '.mp4'
        self.combomoviename = 'combomovie' + my_string + '.mp4'
        self.Udata = None  # the Udata will be called later!
        self.fm = None
        self.fm_error = None
        self.sm = None
        self.sm_error = None

    # a function for actually performing the time-stepping on a simulation object. Adds the property Udata
    # to the simulation object (the actual values of our solution throughout the simulation)
    def run_sim(self, method_kw='etdrk4', print_runtime=True):
        import time
        start = time.time()
        Udata = do_time_stepping(self, method_kw)
        end = time.time()

        self.Udata = Udata

        if print_runtime:
            runtime = end - start
            print('Simulation runtime = %.3f' % runtime, 's')

    # save the simulation object to an external .pkl file using the pickle module.
    def save(self):
        self.model = None # OK this line needs some explaining! Basically the simulation object needs to track its
        # model, and not just the model_kw, bcz during time-stepping we need to of course access the symbol and the
        # forcing term. BUT to make the built-in (non-custom) models easily callable from just the model_kw, we need
        # to define model objects that in turn involve nested functions. And these can't be pickled because of course not
        # why would anything work. So, the best way I found to accommodate all of...
        # 1) having the model available for time-stepping
        # 2) being able to have users define custom models
        # 3) being able to have users just call one of my built-in models with a model_kw
        # was to just forget the actual model attribute of our simulation. Since we still keep the model_kw attribute
        # this is not a big deal when it comes to saving/loading: as long as we have all of the data we have from the sim
        # and the file is named properly, there's nothing to worry about. Said differently, we keep the full model attribute
        # of a simulation object around only as long as we need it.

        my_path = os.path.join("sim_archive")

        # if the archive folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        with open('sim_archive/'+self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # if we know for sure the sim has been done, we can just load it. Since time-stepping only fills the Udata attribute
    # to the simulation object, "loading" a saved sim just means
    # 1) loading the pickle and
    # 2) adding the Udata attribute to our new sim
    def load(self):

        try:

            with open('sim_archive/' + self.filename, 'rb') as inp:
                loaded_sim = pickle.load(inp)
                self.Udata = loaded_sim.Udata

        # due to old filenaming conventions where sometimes what is now the boolean "sponge_layer" was called
        # "absorbing_layer" or "abslayer", we want to allow the possibility of loading old files.
        # TODO: This is only for Adam George Morgan and should be deprecated in future releases/ once the relevant
        #  papers are published

        except:

            old_string = ('_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' +
                            self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '_abslayer='
                            + str(self.sponge_layer))

            old_filename = 'simdata' + old_string + '.pkl'

            with open('sim_archive/' + old_filename, 'rb') as inp:
                loaded_sim = pickle.load(inp)
                self.Udata = loaded_sim.Udata


    # now put everything together: load if possible, but run if you gotta
    def load_or_run(self, method_kw='etdrk4', print_runtime=True, save=True, verbose=True):

        try:
            self.load()
            if verbose:
                print('Saved simulation found, loading saved data.')
            else:
                pass

        except:
            if verbose:
                print('No saved simulation found, running simulation.')
            else:
                pass
            self.run_sim(method_kw=method_kw, print_runtime=print_runtime)

            if save:
                self.save()
            else:
                pass

    # plot the initial state
    def plot_initial_condition(self, custom_ylim = None, dpi=100, color='xkcd:cerulean', fieldname='u', usetex=True,
                           show_figure=True, save_figure=False):

        x = clip_spongeless(self.x, self.sfrac)

        if self.t_ord == 1:
            u = self.initial_state

        elif self.t_ord == 2:
            u = self.initial_state[0,:]

        else:
            raise ValueError('t_ord must be 1 or 2.')

        u = clip_spongeless(u, self.sfrac)

        if usetex:

            try:
                ylabel = r'$u(x,t=0)$'.replace('u', str(fieldname))

                nice_plot(x, u, r'$x$', ylabel, custom_ylim=custom_ylim,
                          show_figure=show_figure, save_figure=save_figure, picname=self.ic_picname,
                          linestyle='solid',
                          color=color, usetex=True, dpi=dpi)

            except RuntimeError:  # catch a user error thinking they have tex when they don't
                usetex = False

        else:

            ylabel = r'u(x,t=0)'.replace('u', str(fieldname))

            nice_plot(x, u, r'x', ylabel, custom_ylim=custom_ylim, show_figure=show_figure,
                      save_figure=save_figure, picname=self.ic_picname,
                      linestyle='solid',
                      color=color, usetex=False, dpi=dpi)


    # create a Hovmoeller plot (filled contour plot in space-time) of the simulation.
    def hov_plot(self, umin=None, umax=None, dpi=100, cmap='cmo.haline', fieldname='u', usetex=True, show_figure=True, save_figure=False):
        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / self.ndump), endpoint=True)

        if self.t_ord == 1:
            u = self.Udata

        elif self.t_ord == 2:
            u = self.Udata[0, :, :]

        else:
            raise ValueError('t_ord can only be 1 or 2.')

        # add right endpoint to prevent a stripe from appearing in the pics
        x_end = np.append(self.x, 0.5 * self.length)
        x_end = clip_spongeless(x_end, self.sfrac)

        if self.complex:
            u_end = 1j*np.zeros((1 + int(self.T / (self.ndump * self.dt)), self.N + 1), dtype=float)

        else:

            u_end = np.zeros((1 + int(self.T / (self.ndump * self.dt)), self.N + 1), dtype=float)

        u_end[:, 0:self.N] = np.copy(u)

        u_end[:, -1] = np.copy(u[:, 0])

        u_end = clip_spongeless(u_end, self.sfrac)

        with spinner('Rendering Hovmoeller plot...'):

            # if the field is real, nothing to do
            if not self.complex:

                hov_plot(x_end, times, u_end, fieldname=fieldname, umin=umin, umax=umax, dpi=dpi, show_figure=show_figure, save_figure=save_figure,
                         picname=self.picname, cmap=cmap, usetex=usetex)

            # if the field is complex, we want to split it into its real and imaginary parts and plot these
            else:

                if usetex:

                    real_fieldname = r'\mathrm{Re}\left(u\right)'.replace('u', str(fieldname))
                    imag_fieldname = r'\mathrm{Im}\left(u\right)'.replace('u', str(fieldname))

                else:

                    real_fieldname = r'Re(u)'.replace('u', str(fieldname))
                    imag_fieldname = r'Im(u)'.replace('u', str(fieldname))

                hov_plot(x_end, times, np.real(u_end), fieldname=real_fieldname, umin=umin, umax=umax, dpi=dpi,
                         show_figure=show_figure, save_figure=save_figure,
                         picname=self.realpicname, cmap=cmap, usetex=usetex)

                hov_plot(x_end, times, np.imag(u_end), fieldname=imag_fieldname, umin=umin, umax=umax, dpi=dpi,
                         show_figure=show_figure, save_figure=save_figure,
                         picname= self.imagpicname, cmap=cmap, usetex=usetex)


    # as above, but just with the modulus of the soln
    def hov_plot_modulus(self, umin=None, umax=None, dpi=100, cmap='cmo.haline', fieldname='u', usetex=True, show_figure=True, save_figure=False):
        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / self.ndump), endpoint=True)

        if self.t_ord == 1:
            u = self.Udata

        elif self.t_ord == 2:
            u = self.Udata[0, :, :]

        else:
            raise ValueError('t_ord can only be 1 or 2.')

        # add right endpoint to prevent a stripe from appearing in the pics
        x_end = np.append(self.x, 0.5 * self.length)
        x_end = clip_spongeless(x_end, self.sfrac)

        if self.complex:

            u_end = 1j*np.zeros((1 + int(self.T / (self.ndump * self.dt)), self.N + 1), dtype=float)

        else:

            u_end = np.zeros((1 + int(self.T / (self.ndump * self.dt)), self.N + 1), dtype=float)

        u_end[:, 0:self.N] = np.copy(u)

        u_end[:, -1] = np.copy(u[:, 0])

        u_end = clip_spongeless(u_end, self.sfrac)

        u_end = np.absolute(u_end)

        with spinner('Rendering Hovmoeller plot of modulus...'):

            if usetex:

                mod_fieldname = r'\left|u\right|'.replace('u', str(fieldname))

            else:

                mod_fieldname = r'|u|'.replace('u', str(fieldname))

            hov_plot(x_end, times, u_end, fieldname=mod_fieldname, umin=umin, umax=umax, dpi=dpi, show_figure=show_figure, save_figure=save_figure,
                         picname=self.modpicname, cmap=cmap, usetex=usetex)


    # save a movie of the evolution of our solution.
    def save_movie(self, fps=200, fieldname='u', usetex=True, fieldcolor='xkcd:ocean green', dpi=100):

        if self.t_ord == 1:
            u = clip_spongeless(self.Udata, self.sfrac)

        elif self.t_ord == 2:
            u = clip_spongeless(self.Udata[0, :, :], self.sfrac)

        else:
            pass

        with spinner('Rendering movie...'):

            if not self.complex:
                save_movie(u, x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                           fieldname=fieldname, fps=fps, ndump=self.ndump, filename=self.moviename,
                           periodic=not self.sponge_layer, usetex=usetex, fieldcolor=fieldcolor, dpi=dpi)

            else:

                if usetex:

                    real_fieldname = r'\mathrm{Re}\left(u\right)'.replace('u', str(fieldname))
                    imag_fieldname = r'\mathrm{Im}\left(u\right)'.replace('u', str(fieldname))

                else:

                    real_fieldname = r'Re(u)'.replace('u', str(fieldname))
                    imag_fieldname = r'Im(u)'.replace('u', str(fieldname))

                save_movie(np.real(u), x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                           fieldname=real_fieldname, fps=fps, ndump=self.ndump, filename=self.realmoviename,
                           periodic=not self.sponge_layer, usetex=usetex, fieldcolor=fieldcolor, dpi=dpi)

                save_movie(np.imag(u), x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                           fieldname=imag_fieldname, fps=fps, ndump=self.ndump, filename=self.imagmoviename,
                           periodic=not self.sponge_layer, usetex=usetex, fieldcolor=fieldcolor, dpi=dpi)


    # save a movie the modulus
    def save_movie_modulus(self, fps=200, fieldname='u', usetex=True, fieldcolor='xkcd:ocean green', dpi=100):

        if self.t_ord == 1:
            u = clip_spongeless(self.Udata, self.sfrac)

        elif self.t_ord == 2:
            u = clip_spongeless(self.Udata[0, :, :], self.sfrac)

        else:
            pass

        u = np.absolute(u)

        with spinner('Rendering movie of modulus...'):

            if usetex:

                mod_fieldname = r'\left|u\right|'.replace('u', str(fieldname))

            else:

                mod_fieldname = r'|u|'.replace('u', str(fieldname))

            save_movie(u, x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                            fieldname=mod_fieldname, fps=fps, ndump=self.ndump, filename=self.modmoviename,
                            periodic=not self.sponge_layer, usetex=usetex, fieldcolor=fieldcolor, dpi=dpi)

    # save a movie of the evolution of our perturbation AND a nested movie of its power spectrum
    def save_combomovie(self, fps=200, fieldname='u', fieldcolor='xkcd:ocean green', speccolor='xkcd:dark orange', usetex=True, dpi=100):
        if self.t_ord == 1:
            u = clip_spongeless(self.Udata, self.sfrac)

        elif self.t_ord == 2:
            u = clip_spongeless(self.Udata[0, :, :], self.sfrac)

        with spinner('Rendering combo movie...'):

            # for a real field there's nothing to do....
            if not self.complex:

                save_combomovie(u,  x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                                fieldname=fieldname, fps=fps, fieldcolor=fieldcolor,
                                speccolor=speccolor, ndump=self.ndump, filename=self.combomoviename,
                                periodic=not self.sponge_layer, usetex=usetex, dpi=dpi)

            # for a complex field just plot the modulus and the power spectrum
            else:

                u = np.absolute(u)

                if usetex:

                    mod_fieldname = r'\left|u\right|'.replace('u', str(fieldname))

                else:

                    mod_fieldname = r'|u|'.replace('u', str(fieldname))

                save_combomovie(u, x=clip_spongeless(self.x, self.sfrac), length=self.length, dt=self.dt,
                                fieldname=mod_fieldname, fps=fps, fieldcolor=fieldcolor,
                                speccolor=speccolor, ndump=self.ndump, filename=self.combomoviename,
                                periodic=not self.sponge_layer, usetex=usetex, dpi=dpi)

    # obtain first moment of the system
    def get_fm(self):

        length = self.length
        N = self.N
        u = self.Udata

        fm = (1./length) * integrate(u, length, N) # take mean

        fm_error = np.abs(fm[1:]-fm[0])

        self.fm = fm
        self.fm_error = fm_error

    # obtain second moment
    def get_sm(self):

        length = self.length
        N = self.N
        u = self.Udata

        if self.fm.any() is None:

            self.get_fm()

        fm = self.fm

        sm = (1./length) * integrate(u**2, length, N)
        sm -= fm**2

        sm_error = np.abs(sm[1:]-sm[0])

        self.sm = sm
        self.sm_error = sm_error

# a function that performs a refinement study based on Richardson extrapolation for error estimation. Very
# useful for quickly checking accuracy. It's here because it doesn't have a better home right now, and it
# *almost* takes simulation objects in as input.
def do_refinement_study(model, initial_state, length, T, Ns, dts, method_kw='etdrk4', bc='periodic', sponge_params=None,
                        show_figure=True, save_figure=False, usetex=True, dpi=400, fit_min=3, fit_max=7):

    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    Ns = Ns.astype(int)
    num_Ns = np.size(Ns)
    num_dts = np.size(dts)

    # initialize outputs

    errors = np.zeros([num_Ns, num_dts], dtype=float)

    cnt = 0

    #start = time.time()
    with spinner('Performing refinement study...'):
        for k in np.arange(0, num_Ns):

            N = Ns[k]

            # do simulation at the worst order (largest time step) first
            rough_st_grid = {'length':length, 'T':T, 'N':N, 'dt':dts[0]}
            rough_sim = simulation(rough_st_grid, model, initial_state, bc=bc, sponge_params=sponge_params)

            rough_sim.load_or_run(method_kw=method_kw, save=True, print_runtime=False, verbose=False)

            for dt in dts:

                fine_st_grid = {'length': length, 'T': T, 'N': N, 'dt': 0.5*dt}
                fine_sim = simulation(fine_st_grid, model, initial_state, bc=bc, sponge_params=sponge_params)

                fine_sim.load_or_run(method_kw=method_kw, save=True, print_runtime=False, verbose=False)

                rough_Udata = rough_sim.Udata

                fine_Udata = fine_sim.Udata

                # use fine sim and rough sim at last time step to get Richardson error estimate

                if fine_sim.t_ord == 1:

                    diff = clip_spongeless(rough_Udata[-1, :] - fine_Udata[-1, :], fine_sim.sfrac)

                elif fine_sim.t_ord == 2:

                    diff = clip_spongeless(rough_Udata[0, -1, :] - fine_Udata[0, -1, :], fine_sim.sfrac)

                else:

                    raise ValueError('Order of temporal derivatives must be 1 or 2')

                errors[k, cnt] = (1. / ((2 ** 4) - 1)) * np.amax(np.absolute(diff))

                rough_sim = fine_sim  # redefine for efficiency... only works bcz we refine dt in powers of 1/2

                cnt += 1

            cnt = 0  # reinit the counter

    #end = time.time()
    #runtime = end - start
    #print('Runtime for accuracy tests = %.4f' % runtime + ' s')

    # now we produce a plot of the errors using an awkward but functioning purpose-built plotting fnc
    plot_refinement_study(model, initial_state, length, T, Ns, dts, errors, method_kw=method_kw, bc=bc,
                        show_figure=show_figure, save_figure=save_figure, usetex=usetex, dpi=dpi)

    # estimate the slope of particular error curves if you want. Needs a bit of by-hand tweaking (controlled by the
    # inputs fit_min, fit_max) bcz for small enough dt we can get level-off or rounding error domination in the error
    # curve, destroying the linear trend after a certain threshold

    params = np.polyfit(np.log10(dts[fit_min:fit_max+1]), np.log10(errors[-1, fit_min:fit_max+1]), 1)
    slope = params[0]
    print('Estimated Slope of Error Line at N = %i' % Ns[-1] + ' is slope = %.3f' % slope)

    return None

def do_refinement_study_alt(model, initial_state, length, T, Ns, dts, benchmark_sim, method_kw='etdrk4', bc='periodic', sponge_params=None, show_figure=True, save_figure=False, usetex=True,
                        fit_min=3, fit_max=7):

    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    Ns = Ns.astype(int)
    num_Ns = np.size(Ns)
    num_dts = np.size(dts)

    # initialize outputs

    errors = np.zeros([num_Ns, num_dts], dtype=float)

    cnt = 0

    benchmark_sim.load_or_run(method_kw=method_kw, save=True, print_runtime=False, verbose=False)

    #start = time.time()
    with spinner('Performing refinement study...'):
        for k in np.arange(0, num_Ns):

            N = Ns[k]

            for dt in dts:

                stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
                rough_sim = simulation(stgrid, model, initial_state, bc=bc, sponge_params=sponge_params)

                rough_sim.load_or_run(method_kw=method_kw, save=True, print_runtime=False, verbose=False)

                rough_Udata = rough_sim.Udata

                benchmark_Udata = benchmark_sim.Udata

                # use fine sim and rough sim at last time step to get Richardson error estimate

                diff = clip_spongeless(rough_Udata[-1, :]-benchmark_Udata[-1, :], benchmark_sim.sfrac)

                errors[k, cnt] = np.amax(np.abs(diff))

                cnt += 1

            cnt = 0  # reinit the counter

    #end = time.time()
    #runtime = end - start
    #print('Runtime for accuracy tests = %.4f' % runtime + ' s')

    # now we produce a plot of the errors
    fig, ax = plt.subplots()

    dts = 0.5 * dts

    # define the cycler
    my_cycler = (
                plt.cycler(color=['xkcd:slate', 'xkcd:raspberry', 'xkcd:goldenrod', 'xkcd:deep green'])
                + plt.cycler(lw=[3.5, 3, 2.5, 2])
                + plt.cycler(linestyle=['dotted', 'dashed', 'solid', 'dashdot'])
                + plt.cycler(marker=['v', '*', 'o', 'P'])
                + plt.cycler(markersize=[8, 12, 8, 8])
    )

    ax.set_prop_cycle(my_cycler)

    for m in range(0, num_Ns):
        if usetex:
            plt.loglog(dts, errors[m, :], label=r'$N = z$'.replace('z', str(Ns[m])))
        # ^ an awesome trick from
        # https://stackoverflow.com/questions/33786332/matplotlib-using-variables-in-latex-expressions
        # was used to get the labels working as above
        else:
            plt.loglog(dts, errors[m, :], label='N = z'.replace('z', str(Ns[m])))

    ax.legend(fontsize=16)

    if usetex:
        plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
        plt.ylabel(r"Absolute Error", fontsize=26, color='k')
    else:
        plt.xlabel("Δt", fontsize=26, color='k')
        plt.ylabel("Absolute Error", fontsize=26, color='k')

    plt.tick_params(axis='x', which='both', top='off', color='k')
    plt.xticks(fontsize=16, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right='off', color='k')
    plt.yticks(fontsize=16, rotation=0, color='k')

    plt.tight_layout()

    if save_figure is True:

        # add the folder "visuals" to our path
        my_path = os.path.join("visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and now we can save the fig
        if bc == 'sponge_layer':
            sponge_layer = True
        elif bc == 'periodic':
            sponge_layer = False

        my_string = ('_length=%.1f_T=%.1f' % (
        length, T) + '_modelkw=' + model.model_kw + '_ICkw=' + initial_state.initial_state_kw + '_method_kw='
                     + method_kw + '_nonlinear=' + str(model.nonlinear) + '_abslayer=' + str(sponge_layer))

        picname = 'refinement_study' + my_string + '.png'
        plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=400)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

    # estimate the slope of particular error curves if you want. Needs a bit of by-hand tweaking (controlled by the
    # inputs fit_min, fit_max) bcz for small enough dt we can get level-off or rounding error domination in the error
    # curve, destroying the linear trend after a certain threshold

    params = np.polyfit(np.log10(dts[fit_min:fit_max+1]), np.log10(errors[-1, fit_min:fit_max+1]), 1)
    slope = params[0]
    print('Estimated Slope of Error Line at N = %i' % Ns[-1] + ' is slope = %.3f' % slope)

    return None