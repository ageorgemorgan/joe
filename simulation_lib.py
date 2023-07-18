import pickle

import numpy as np

import matplotlib.pyplot as plt

import cmocean.cm as cmo

from time_stepper import do_time_stepping

from initial_states import initial_state

from visualization import hov_plot, save_movie, save_combomovie

# a class for simulations. You init with the physical basics and discretization parameters,
# then you can call a run simulation function on a simulation object


class simulation:
    def __init__(self, length, T, N, dt, model_kw, initial_state_kw, nonlinear=True, absorbing_layer=False, ndump=10):
        self.length = length
        self.T = T
        self.N = N
        self.dt = dt
        self.model_kw = model_kw
        self.initial_state_kw = initial_state_kw
        self.nonlinear = nonlinear
        self.absorbing_layer = absorbing_layer
        self.ndump = ndump # hyperparameter describing how often we save our time steps
        self.x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.N, endpoint=False)
        self.initial_state = initial_state(self.x, self.initial_state_kw)
        self.filename = 'simdata_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '_abslayer=' + str(self.absorbing_layer) + '.pkl'
        self.picname = 'hovplot_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '_abslayer=' + str(self.absorbing_layer)+ '.png'
        self.moviename = 'movie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '_abslayer=' + str(self.absorbing_layer)+ '.mp4'
        self.combomoviename = 'combomovie_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (self.length, self.T, self.N, self.dt) + '_modelkw=' + self.model_kw + '_ICkw=' + self.initial_state_kw + '_nonlinear=' + str(self.nonlinear) + '_abslayer=' + str(self.absorbing_layer)+ '.mp4'
        self.Udata = None  # the Udata will be called later!

    # a function for actually performing the time-stepping on a simulation object. Adds the property Udata
    # to the simulation object (the actual values of our solution throughout the simulation)
    def run_sim(self):

        Udata = do_time_stepping(self)

        self.Udata = Udata

    # save the simulation object to an external .pkl file using the pickle module.
    def save(self):

        with open(self.filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    # create a Hovmoeller plot (filled contour plot in space-time) of the simulation.
    def hov_plot(self, show_figure=True, save_figure=False):

        nsteps = int(self.T/self.dt)
        times = np.linspace(0., self.T, num=1+int(nsteps / self.ndump), endpoint=True)

        u = self.Udata
        # u = self.Udata[0, :, :]

        hov_plot(self.x, times, u, fieldname='$u(x,t)$', show_figure=show_figure, save_figure=save_figure, picname=self.picname, cmap=cmo.haline)

    # save a movie of the evolution of our solution.
    def save_movie(self):

        u = self.Udata

        # u = self.Udata[0, :, :]

        save_movie(u, x=self.x, length=self.length, dt=self.dt, ndump=self.ndump, filename=self.moviename, periodic=True)

    # save a movie of the evolution of our perturbation AND a nested movie of its power spectrum
    def save_combomovie(self):

        u = self.Udata

        # u = self.Udata[0, :, :]

        save_combomovie(u, x=self.x, length=self.length, dt=self.dt, ndump=self.ndump, filename=self.combomoviename)