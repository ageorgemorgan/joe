import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from .time_stepper import do_time_stepping
from .visualization import hov_plot, save_movie, save_combomovie, spinner, plot_refinement_study, nice_plot
from .sponge_layer import clip_spongeless
from .utils import integrate

class model:
    r"""A class for models, which in *joe* specifically refers to particular PDEs.

    This defines a model PDE with the name 'model_kw' in the Fourier-space form

    .. math::
        \left(\partial_t\right)^{\texttt{t\_ord}} V(k,t) + \texttt{symbol}(k) V(k,t) = \texttt{fourier\_forcing}

    where :math:`V(k,t)` is the Fourier state.

    Note how the init takes in two callables for the symbol and forcing terms, but these get converted to dicts.
    This is to avoid making the weird no-no of having callable attributes. The trick comes from
    <https://stackoverflow.com/questions/35321744/python-function-as-class-attribute-becomes-a-bound-method>... we
    instead make the "callable attributes" !dicts! with callable entries. This dirty trick is all under the hood in
    the model class, so when defining and using a model object the user doesn't need to care.

    Attributes
    ----------
        model_kw : str
            A name for the model.
        t_ord : int
            Temporal order of derivatives in the model. Currently only t_ord=1,2 are supported.
        symbol : dict
            Symbol of the linear, constant-coefficient part of the PDE. symbol['symbol'] must be a callable with a
            single argument `k`, representing a Fourier-space frequency variable.
        fourier_forcing : dict
            Forcing term in the PDE, represented in Fourier space. fourier_forcing['fourier_forcing'] must be a callable
            that takes in four arguments `(V, k, x, nonlinear)` where `V,k,x` are ndarrays and `nonlinear` is a boolean.
        nonlinear : boolean, optional
            True if we include nonlinearity in the model, False otherwise. Default: True.
        complex : boolean
            False if solution to the PDE is supposed to be real, True otherwise. Default: False.
    """
    def __init__(self, model_kw, t_ord, symbol, fourier_forcing, nonlinear=True, complex=False):
        self.model_kw = model_kw
        self.t_ord = t_ord  # an integer
        self.symbol = {'symbol': symbol}  # callable
        self.fourier_forcing = {'fourier_forcing': fourier_forcing}  # callable
        self.nonlinear = nonlinear
        self.complex = complex

    def get_symbol(self, *args):
        return self.symbol['symbol'](*args)

    # obtain the forcing term in Fourier space
    def get_fourier_forcing(self, *args):
        return self.fourier_forcing['fourier_forcing'](*args)

class initial_state:
    r"""A class for initial states.

    Having a special class for initial states instead of just representing them as callables or keywords allows for
    greater flexibility, as well as some symmetry with the `model` class :class:`~joe_lab.joe.model`.

    Attributes
    ----------
        initial_state_kw : str
            Name of the initial state.
        initial_state_func : dict
            A dict defined such that initial_state_func['initial_state_func'] is a callable representing the actual
            initial state function. One must be careful of this function's output's shape when dealing with second-order
            problems.
        initial_state : ndarray
            The values of the initial state on our particular grid. Not defined during init, and should only be accessed
            under-the-hood.
    """
    def __init__(self, initial_state_kw, initial_state_func):
        self.initial_state_kw = initial_state_kw
        self.initial_state_func = {'initial_state_func': initial_state_func}
        self.initial_state = None

    # obtain the actual initial state fnc
    def get_initial_state(self, *args):
        return self.initial_state_func['initial_state_func'](*args)


# a class for simulations.

class simulation:
    r"""A class for simulations.

    This class is the heart of *joe*, and most of the scientific information (including visualizations) can be accessed
    via the attributes of methods of a simulation instance.

    You init an instance with an `stgrid` (space-time grid) dict, an instance of the `model` class (:class:`~joe_lab.joe.model`) to
    specify the PDE, and an instance of the `initial_state` class (:class:`~joe_lab.joe.initial_state`). Then you can
    call a run simulation function on a simulation object to solve the IVP and store/save the solution, or load up the
    solution from an experiment you've already done.

    Attributes
    ----------
        length : float
            Length of the spatial domain.
        T : float
            Total physical runtime of the simulation. That is, the spacetime grid covers the time interval [0,T].
        N : int
            Number of spatial locations on which to sample the solution to our PDE.
        dt : float
            Time step size for numerical integration of the PDE.
        model : model
            An instance of the model class, see :class:`~joe_lab.joe.model`.
        model_kw : str
            A name for the model.
        t_ord : int
            Temporal order of derivatives in the model. Currently only `t_ord=1,2` are supported.
        initial_state_kw : str
            Name of the initial state.
        ndump : int
            Only every `ndump` timesteps are stored when saving simulation results.
        Udata : ndarray
            Values of the solution to our PDE sampled on our space-time grid, stored as an array. The 0 axis represents
            temporal variation, and the 1 axis represents spatial variation (so each row is a fixed-time snapshot of
            the solution on our spatial grid).
        nonlinear : boolean
            True if we include nonlinearity in the model, False otherwise. Default: True.
        complex : boolean
            False if solution to the PDE is supposed to be real, True otherwise. Default: False.
        sponge_params : dict
            Contains particular parameters for the sponge layer, see :func:`~joe_lab.sponge_layer.damping_coeff_lt`.
        sponge_layer : boolean
            True if sponge layer is included, False otherwise. Default: False.
        sfrac : float
            Fraction of the spatial grid that is not taken up by the sponge layer. By convention, this fraction is taken
            from the middle of the grid.
        x : ndarray
            Spatial grid of [-0.5*`length`, 0.5*`length`].
        initial_state : ndarray
            Initial state function sampled at `x`
        fm : ndarray
            First moments (spatial integrals) of the solution to our PDE.
        fm_error : ndarray
            Absolute difference between `fm` and `fm[0]`, the initial first moment.
        sm : ndarray
            Second moments (spatial L2 - norms) of the solution to our PDE.
        sm_error : ndarray
            Absolute difference between `sm` and `sm[0]`, the initial second moment.
        filename : str
            Name of the .pkl file the sim may be saved in= 'ic_plot, or loaded from.
        ic_picname : str
            Name of the initial state plot as a .png file.
        picname : str
            Name of the Hovmoeller (filled space-time contour) plot of the solution as a .png file.
        realpicname : str
            Name of the Hovmoeller plot of the real part of the solution as a .png file.
        imagpicname : str
            Name of the Hovmoeller plot of the imaginary part of the solution as a .png file.
        modpicname : str
            Name of the Hovmoeller plot of the modulus of the solution as a .png file.
        moviename : str
            Name of the movie of the solution as a .mp4 file.
        realmoviename : str
            Name of the movie of the real part of the solution as a .mp4 file.
        imagmoviename : str
            Name of the movie of the imaginary part of the solution as a .mp4 file.
        modmoviename : str
            Name of the movie of the modulus of the solution as a .mp4 file.
        combomoviename : str
            Name of the movie of the solution (or its modulus if `complex=True`) and its power spectrum as a .mp4 file.
    """
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

        my_string = ('_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                       self.dt) + '_modelkw=' + self.model_kw
                     + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_sponge_layer=' + str(self.sponge_layer))

        self.filename = 'simdata' + my_string + '.pkl'
        self.ic_picname = 'ic_plot' + my_string + '.png'
        self.ic_imag_picname = 'ic_plot_imag' + my_string + '.png'
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
        self.fm = None # first & second moments, error in these will likewise be called later
        self.fm_error = None
        self.sm = None
        self.sm_error = None

    def run_sim(self, method_kw='etdrk4', print_runtime=True):
        r"""Perform the time-stepping starting from the initial state and ending at time `T`. Populates the attribute
        `Udata` (the actual values of our solution throughout the simulation) in our `simulation`.

        Parameters
        ----------
        method_kw : str, optional
            Name of the numerical method. Currently, 'etdrk1', 'ifrk4' are available for first-order-in-time problems,
            and 'etdrk4' is available for all problems: see :class:`~joe_lab.time_stepper.time_stepper`.
            Default: 'etdrk4'.
        print_runtime: boolean, optional
            True if you would like to print the wall-clock runtime of the simulation, False otherwise.
        """
        import time
        start = time.time()
        Udata = do_time_stepping(self, method_kw)
        end = time.time()

        self.Udata = Udata

        if print_runtime:
            runtime = end - start
            print('Simulation runtime = %.3f' % runtime, 's')


    def save(self):
        r"""Save the simulation object to an external .pkl file in the joe_sim_archive directory using the pickle module.

        For a technical python reason, calling this function reassigns the simulation's `model` attribute to None.
        """
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

        my_path = os.path.join("joe_sim_archive")

        # if the archive folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        with open('joe_sim_archive/'+self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load(self):
        r"""Loads a sim from a .pkl file that already exists in the joe_sim_archive directory.

        Since time-stepping only fills the `Udata` attribute to the `simulation` instance, "loading" a saved sim just means

        1) loading the pickle and

        2) adding the `Udata` attribute to our `simulation` instance.
        """
        try:

            with open('joe_sim_archive/' + self.filename, 'rb') as inp:
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

            with open('joe_sim_archive/' + old_filename, 'rb') as inp:
                loaded_sim = pickle.load(inp)
                self.Udata = loaded_sim.Udata

    def load_or_run(self, method_kw='etdrk4', print_runtime=True, save=True, verbose=True):
        r"""Load our simulation from the joe_sim_archive directory if it's there, or run the simulation if it can't be
        found in joe_sim_archive. The end result is that our simulation's `Udata` attribute becomes populated with the
        space-time grid values of our solution.

        I recommend you use this method as your on-the-ground alternative to `run_sim`.

        Parameters
        ----------
         method_kw : str, optional
            Name of the numerical method. Currently, 'etdrk1', 'ifrk4' are available for first-order-in-time problems,
            and 'etdrk4' is available for all problems: see :class:`~joe_lab.time_stepper.time_stepper`.
            Default: 'etdrk4'.
        print_runtime: boolean, optional
            True if you would like to print the wall-clock runtime of the simulation, False otherwise.
        save : boolean, optional
            True if you definitely want the simulation saved, and False otherwise. Default: True.
        verbose : boolean, optional
            True if you want to see printed messages about whether a saved simulation was found in the
            joe_sim_archive directory.
        """
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

    def plot_initial_condition(self, custom_ylim = None, dpi=100, color='xkcd:cerulean', fieldname='u', usetex=True,
                           show_figure=True, save_figure=False):
        r"""Produce a line plot of the initial state.

        Parameters
        ----------
        custom_ylim : array_like, optional.
            Custom values for the y-limits on the axes. Default: None.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        color : str, optional
            Name of the color you want the curve to be. For a list of great colors see <https://xkcd.com/color/rgb/>.
            Default: 'xkcd:cerulean'.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at all.
        """
        x = clip_spongeless(self.x, self.sfrac)

        if self.t_ord == 1:
            u = self.initial_state

        elif self.t_ord == 2:

            try:
                u = self.initial_state[0,:]
            except IndexError:
                u = self.initial_state # this line here is to catch the weird case where you're second order
                # but only prescribed one bc on u, not u_t. So when starting the time-stepper joe automatically assumes
                # the inital u_t is zero.

        else:
            raise ValueError('t_ord must be 1 or 2.')

        u = clip_spongeless(u, self.sfrac)

        # if the field is complex, we want to split it into its real and imaginary parts and plot these
        if self.complex:

            if usetex:

                try:
                    real_ylabel = r'\mathrm{Re}\left(u(x,0)\right)'.replace('u', str(fieldname))
                    imag_ylabel = r'\mathrm{Im}\left(u(x,0)\right)'.replace('u', str(fieldname))
                    xlabel = r'$x$'

                except RuntimeError:
                    usetex = False

            else:

                real_ylabel = r'Re(u(x,0))'.replace('u', str(fieldname))
                imag_ylabel = r'Im(u(x,0))'.replace('u', str(fieldname))
                xlabel = r'x'

            nice_plot(x, np.real(u), xlabel, real_ylabel, custom_ylim=custom_ylim,
                      show_figure=show_figure, save_figure=save_figure, picname=self.ic_picname,
                      linestyle='solid',
                      color=color, usetex=True, dpi=dpi)

            nice_plot(x, np.imag(u), xlabel, imag_ylabel, custom_ylim=custom_ylim,
                      show_figure=show_figure, save_figure=save_figure, picname=self.ic_imag_picname,
                      linestyle='solid', color=color, usetex=True, dpi=dpi)

        else:

            if usetex:

                try:
                    ylabel = r'$u(x,t=0)$'.replace('u', str(fieldname))
                    xlabel = r'$x$'

                except RuntimeError:  # catch a user error thinking they have tex when they don't
                    usetex = False

            else:
                ylabel = r'u(x,t=0)'.replace('u', str(fieldname))
                xlabel = r'x'

            nice_plot(x, u, xlabel, ylabel, custom_ylim=custom_ylim,
                      show_figure=show_figure, save_figure=save_figure, picname=self.ic_picname,
                      linestyle='solid', color=color, usetex=True, dpi=dpi)

    def hov_plot(self, umin=None, umax=None, dpi=100, cmap='cmo.haline', fieldname='u', usetex=True, show_figure=True,
                 save_figure=False):
        r"""Create a Hovmoeller plot (filled contour plot in space-time) of the PDE solution.

        If the PDE solution is complex-valued, this produces Hovmoeller plots of its real part and its imaginary part.

        Parameters
        ----------
        umin : float, optional.
            Minimum field value to include, used to construct colorbar lower limit. Default: None.
        umax : float, optional.
            Maximum field value to include, used to construct colorbar upper limit. Default: None.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        cmap : str
            Name of the colormap to use. For a list of great colormaps see <https://matplotlib.org/cmocean/>.
            Default: 'cmo.haline'.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at all.
            Default: False.
        """
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

                hov_plot(x_end, times, u_end, fieldname=fieldname, umin=umin, umax=umax, dpi=dpi,
                         show_figure=show_figure, save_figure=save_figure,
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


    def hov_plot_modulus(self, umin=None, umax=None, dpi=100, cmap='cmo.haline', fieldname='u', usetex=True,
                         show_figure=True, save_figure=False):
        r"""The same as :func:`~joe_lab.joe.hov_plot`, but with the modulus of the field.

        Parameters
        ----------
        umin : float, optional.
            Minimum field value to include, used to construct colorbar lower limit. Default: None.
        umax : float, optional.
            Maximum field value to include, used to construct colorbar upper limit. Default: None.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        cmap : str
            Name of the colormap to use. For a list of great colormaps see <https://matplotlib.org/cmocean/>.
            Default: 'cmo.haline'.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
            all. Default: False.
        """
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

            hov_plot(x_end, times, u_end, fieldname=mod_fieldname, umin=umin, umax=umax, dpi=dpi,
                     show_figure=show_figure, save_figure=save_figure,
                     picname=self.modpicname, cmap=cmap, usetex=usetex)

    def save_movie(self, fps=200, fieldname='u', usetex=True, fieldcolor='xkcd:ocean green', dpi=100):
        r"""Save a movie of the evolution of our solution as a .mp4 file.

        If the PDE solution is complex-valued this produces movies of its real part and its imaginary part.

        Parameters
        ----------
        fps: int, optional
            Frames-per-second for the movie. Default: 200.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        fieldcolor: str, optional
            Name of the color you want the curve to be. For a list of great colors see <https://xkcd.com/color/rgb/>.
            Default: 'xkcd:ocean green'.
        """

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
        r"""The same as :func:`~joe_lab.joe.save_movie`, but with the modulus of the field plotted instead.

        Parameters
        ----------
        fps: int, optional
            Frames-per-second for the movie. Default: 200.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        fieldcolor: str, optional
            Name of the color you want the curve to be. For a list of great colors see <https://xkcd.com/color/rgb/>.
            Default: 'xkcd:ocean green'.
        """
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

    def save_combomovie(self, fps=200, fieldname='u', fieldcolor='xkcd:ocean green', speccolor='xkcd:dark orange',
                        usetex=True, dpi=100):
        r"""Save a movie of the evolution of our field as well as a nested, tinier movie of its power spectrum.

        Parameters
        ----------
        fps: int, optional
            Frames-per-second for the movie. Default: 200.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        fieldname : str, optional
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        fieldcolor: str, optional
            Name of the color you want the curve to be. For a list of great colors see <https://xkcd.com/color/rgb/>.
            Default: 'xkcd:ocean green'.
        speccolor: str, optional
            Name of the color you want the curve of the power spectrum to be. Default: 'xkcd:dark orange'.
        """
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

    def get_fm(self):
        r"""Get the first moment (spatial integral) of the solution to our PDE, `fm`.

        Also gets the absolute difference between `fm` and `fm[0]`, the initial first moment.
        """
        length = self.length
        N = self.N
        u = self.Udata

        fm = (1./length) * integrate(u, length) # take mean

        fm_error = np.abs(fm[1:]-fm[0])

        self.fm = fm
        self.fm_error = fm_error

    # obtain second moment
    def get_sm(self):
        r"""Get the second moment (spatial L2 - norm) of the solution to our PDE, `sm`.

        Also gets the absolute difference between `sm` and `sm[0]`, the initial second moment.
        """
        length = self.length
        N = self.N
        u = self.Udata

        if self.fm.any() is None:

            self.get_fm()

        fm = self.fm

        sm = (1./length) * integrate(np.absolute(u) ** 2, length) # np.absolute for complex-valued fields
        sm -= fm**2

        sm_error = np.abs(sm[1:]-sm[0])

        self.sm = sm
        self.sm_error = sm_error

# TODO: the func below here because it doesn't have a better home right now, and it *almost* takes simulation
#     objects in as input.
def do_refinement_study(model, initial_state, length, T, Ns, dts, method_kw='etdrk4', bc='periodic', sponge_params=None,
                        show_figure=True, save_figure=False, usetex=True, dpi=400, fit_min=3, fit_max=7):
    r"""Performs a refinement study based on Richardson extrapolation for error estimation. Very useful for quickly
    and painless checking accuracy.

    Parameters
    ----------
    model: model
        Instance of the class `model` :class:`~joe_lab.joe.model`.
    initial_state: initial_state
        Instance of the class `initial state` :class:`~joe_lab.joe.initial_state`.
    length : float
        Length of the spatial domain
    T : float
        Total physical runtime of the simulation. That is, the spacetime grid covers the time interval [0,T].
    Ns : list of ints
        Number of spatial locations on which to sample the solution to our PDE.
    dts : list of floats
        Time step sizes for numerical integration of the PDE.
    method_kw : str, optional
        Name of the numerical method. Currently, 'etdrk1', 'ifrk4' are available for first-order-in-time problems,
        and 'etdrk4' is available for all problems: see :class:`~joe_lab.time_stepper.time_stepper`.
        Default: 'etdrk4'.
    bc : str, optional
        String specifying the boundary conditions. Must be 'periodic' or 'sponge_layer'.
    sponge_params: dict
        Contains particular parameters for the sponge layer, see :func:`~joe_lab.sponge_layer.damping_coeff_lt`.
    dpi : int, optional
        Dots-per-inch on the image. Default: 400.
    usetex : boolean, optional
        True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
    show_figure: boolean, optional
        True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
    save_figure: boolean, optional
        True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
        all. Default: False.
    fit_min: int, optional.
        Index of dts at which we *start* pulling values from the Ns[-1] error curve to approximate slope of error curve.
        Default: 3.
    fit_max: int, optional
        Index of dts at which we *stop* pulling values from the Ns[-1] error curve to approximate slope of error curve.
        Default: 7.
    """
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

    return slope


# TODO: this function below needs to be integrated into the other error test better, or scrubbed from the package!!!
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

        # add the folder "joe_visuals" to our path
        my_path = os.path.join("joe_visuals")

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
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=400)

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
