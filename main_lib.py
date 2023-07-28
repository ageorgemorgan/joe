import pickle

import numpy as np

import matplotlib.pyplot as plt

import cmocean.cm as cmo

from time_stepper import do_time_stepping

from initial_states import initial_state

from visualization import hov_plot, save_movie, save_combomovie


# a class for models. Note how the init takes in two callables for the symbol and forcing terms: to avoid making the
# weird no-no of having callable attributes, we use a trick from
# https://stackoverflow.com/questions/35321744/python-function-as-class-attribute-becomes-a-bound-method
# and instead make the "callable attributes" !dicts! with callable entries. This dirty trick is all under the hood
# in the model class, so when defining and using a model object the user doesn't need to care.
class model:
    def __init__(self, model_kw, t_ord, symbol, fourier_forcing, nonlinear=True):
        self.model_kw = model_kw
        self.t_ord = t_ord  # an integer
        self.symbol = {'symbol': symbol}  # callable
        self.fourier_forcing = {'fourier_forcing': fourier_forcing} # callable
        self.nonlinear = nonlinear

        # this defines a model PDE with the name 'model_kw' in the Fourier-space form

        # (d_t)^{t_ord} V + symbol*V = fourier_forcing

        # where V is the Fourier state.

    # obtain the actual symbol on a given mesh
    def get_symbol(self, *args):
        return self.symbol['symbol'](*args)

    # obtain the forcing term in Fourier space
    def get_fourier_forcing(self, *args):
        return self.fourier_forcing['fourier_forcing'](*args)


# turn initial states into classes as well, following the syntax used to define the model class. This is really there
# instead of kw's only just to allow the user to easy input custom initial states on the fly.
class initial_state:
    def __init__(self, initial_state_kw, initial_state_func):
        self.initial_state_kw = initial_state_kw
        self.initial_state_func = {'initial_state_func': initial_state_func}
        self.initial_state = None

    # obtain the actual symbol on a given mesh
    def get_initial_state(self, *args):
        return self.initial_state_func['initial_state_func'](*args)


# a class for simulations. You init with a "model" object to specify the PDE, an "initial_state" object, and discretization parameters.
# Then you can call a run simulation function on a simulation object to solve the IVP and store/save the soln.
class simulation:
    def __init__(self, length, T, N, dt, model, initial_state, absorbing_layer=False, ndump=10):
        self.length = length
        self.T = T
        self.N = N
        self.dt = dt
        self.model = model  # a model object
        self.model_kw = model.model_kw
        self.t_ord = model.t_ord  # an integer
        self.initial_state_kw = initial_state.initial_state_kw
        self.nonlinear = model.nonlinear
        self.absorbing_layer = absorbing_layer
        self.ndump = ndump  # hyperparameter describing how often we save our time steps
        self.x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.N, endpoint=False)
        self.initial_state = initial_state.get_initial_state(self.x) # IMPORTANT: self.initial_state is an array, but the
        # actual input initial_state to the simulation class is an initial_state object!
        self.filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                       self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_abslayer=' + str(self.absorbing_layer) + '.pkl'
        self.picname = 'hovplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                      self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_abslayer=' + str(self.absorbing_layer) + '.png'
        self.moviename = 'movie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                      self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_abslayer=' + str(self.absorbing_layer) + '.mp4'
        self.combomoviename = 'combomovie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N,
                                                                                self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(
            self.nonlinear) + '_abslayer=' + str(self.absorbing_layer) + '.mp4'
        self.Udata = None  # the Udata will be called later!

    # a function for actually performing the time-stepping on a simulation object. Adds the property Udata
    # to the simulation object (the actual values of our solution throughout the simulation)
    def run_sim(self, method_kw='etdrk4', splitting_method_kw='naive'):
        Udata = do_time_stepping(self, method_kw, splitting_method_kw)

        self.Udata = Udata

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
        with open(self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # create a Hovmoeller plot (filled contour plot in space-time) of the simulation.
    def hov_plot(self, colourmap='cmo.haline', show_figure=True, save_figure=False):
        nsteps = int(self.T / self.dt)
        times = np.linspace(0., self.T, num=1 + int(nsteps / self.ndump), endpoint=True)

        if self.t_ord == 1:
            u = self.Udata

        elif self.t_ord == 2:
            u = self.Udata[0, :, :]

        hov_plot(self.x, times, u, fieldname='$u(x,t)$', show_figure=show_figure, save_figure=save_figure,
                 picname=self.picname, cmap=colourmap)

    # save a movie of the evolution of our solution.
    def save_movie(self):

        if self.t_ord == 1:
            u = self.Udata

        elif self.t_ord == 2:
            u = self.Udata[0, :, :]

        save_movie(u, x=self.x, length=self.length, dt=self.dt, ndump=self.ndump, filename=self.moviename,
                   periodic=True)

    # save a movie of the evolution of our perturbation AND a nested movie of its power spectrum
    def save_combomovie(self):
        if self.t_ord == 1:
            u = self.Udata

        elif self.t_ord == 2:
            u = self.Udata[0, :, :]

        save_combomovie(u, x=self.x, length=self.length, dt=self.dt, ndump=self.ndump, filename=self.combomoviename)
