import os

import numpy as np

import matplotlib.pyplot as plt

import cmocean.cm as cmo

import matplotlib.animation as animation

from scipy.interpolate import CubicSpline

from numpy.fft import fft, fftfreq, fftshift

import sys


# create Hovmoeller plot of a scalar field u


def hov_plot(x, t, u, fieldname, show_figure=True, save_figure=False, picname="", cmap=cmo.haline):
    # I like to use tex to render the graph labels nicely. However, for the benefit of users who either don't
    # have tex installed or are encountering some issue where matplotlib doesn't want to talk to tex,
    # I need an "out" ... TODO: this!

    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = "serif"

    umin = -0.8#np.amin(u)
    umax = 1.1#np.amax(u)
    levels = np.linspace(umin, umax, num=300)

    CF = plt.contourf(x, t, u, cmap=cmap, levels=levels)

    # axis labels
    plt.xlabel(r"$x$", fontsize=22, color='k')
    plt.ylabel(r"$t$", fontsize=22, color='k')

    plt.tick_params(axis='x', which='both', top=False, color='k')
    plt.xticks(fontsize=16, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right=False, color='k')
    plt.yticks(fontsize=16, rotation=0, color='k')

    plt.xlim([-25, 100])

    # make colorbar
    cbar = plt.colorbar(CF, format='%.2f')
    cbar.ax.tick_params(labelsize=16, color='k')
    plt.clim(umin, umax)
    cbar.ax.set_ylabel(fieldname, fontsize=22, color='k')

    # the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
    # This is not so easy, and relies on the thread at
    # https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='k')

    plt.tight_layout()

    if save_figure is True:

        # add the folder "visuals" to our path... more on this below
        my_path = os.path.join("visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and save the fig
        plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=800)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass


# create a movie from a scalar field u(t,x) sampled at various times.


def save_movie(u, x, length, dt, ndump, filename, periodic=True, dpi=100):
    # Create movie file in mp4 format. Warning: this is very slow!

    # TODO: I think the "no_tex" killswitch isn't working

    no_tex = False

    try:
        plt.rc('text', usetex=True)

    except RuntimeError:
        plt.rcParams["font.family"] = "serif"
        no_tex = True

    fig = plt.figure()

    umin = 1.05 * np.amin(u)
    umax = 1.05 * np.amax(u)

    #from scipy.integrate import simps

    #coeff = simps(np.abs(u[0, :]),x)  #TODO: multiplying by a factor of 4 gives a solid result , but for BBM 6 is # the
    # correct "back of the envelope" value.

    # TODO: CAREFUL THE xLIM HERE!!!!! MAKE A FUNCTION OF ABSORBING LAYER
    ax = plt.axes(xlim=(-0.5 * length, 0.5 * length), ylim=(umin, umax))
    #ax = plt.axes(xlim=(-60, 25), ylim=(umin, umax))

    fig.set_size_inches(8, 6)

    # use cubic spline interpolation to smooth the data

    # note that spline interpolation is necessary to combat the Runge phenomenon since we have a uniform grid! Naive
    # use of barycentric interpolation is not an option.

    if periodic:

        # add endpoint
        x_end = np.append(x, 0.5 * length)

        [M, N] = np.shape(u)

        u_end = np.zeros([M, N + 1], dtype=float)

        u_end[:, 0:N] = u

        u_end[:, -1] = np.copy(u[:, 0])  # enforce periodicity

        # create the interpolating spline polynomial using scipy interpolation package

        poly = CubicSpline(x_end, u_end, axis=1, bc_type='periodic')

    else:

        poly = CubicSpline(x, u, axis=1)

    # now we can actually do the upsampling

    NN = 1200  # number of points to evaluate interpolant at

    xx = np.linspace(-0.5 * length, 0.5 * length, NN, endpoint=True)

    # uu = poly(xx)

    # x = xx

    # u = uu

    ax.grid('True')

    if periodic:

        # color = 'xkcd:ocean green'
        color = 'xkcd:cerulean'

    else:

        color = 'xkcd:dark magenta'

    line, = ax.plot([], [], linewidth=2, color=color, label='$u(t,x)$')
    #bound_line_up, = ax.plot([], [], linewidth=2, color='xkcd:emerald', linestyle='dashed',
                            # label='$\|u_0\|_{L^{1}_{x}} (1+t)^{-1/3}$')
    #bound_line_down, = ax.plot([], [], linewidth=2, color='xkcd:emerald', linestyle='dashed')

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(u, ax)
    timer.start()

    #ax.legend(fontsize=20, loc='upper left')

    plt.tick_params(axis='x', which='both', top=False, color='k')
    plt.xticks(fontsize=20, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right=False, color='k')
    plt.yticks(fontsize=20, rotation=0, color='k')

    plt.xlabel(r"$x$", fontsize=22, color='k')


    # Animation function.
    def animate(i):
        line.set_data(x, u[i, :])

        tplot = i * dt * ndump

        #bound_line_up.set_data(x, coeff / ((1 + tplot) ** (1. / 3.)))
        #bound_line_down.set_data(x, -coeff / ((1 + tplot) ** (1. / 3.)))

        #plt.title('$t=%.2f$' % tplot, fontsize=22)

        if no_tex:
            ylabel_str= 'u(x,t=%.2f)' % tplot

        else:
            ylabel_str= r'$u(x,t=%.2f)$' % tplot

        ax.set_ylabel(ylabel_str, fontsize=22)
        """

        if periodic:

            ax.set_ylabel('$u(x,t=%.2f)$' % tplot, fontsize=22)

        else:

            ax.set_ylabel('$\phi(x,t=%.2f)$' % tplot, fontsize=22)
            
        """

        # TODO: the above label fixes are kind of a jerry-rigged solution and don't generalize easily to other PDE!
        #     Find a smart way to fix this!

        plt.xlim([-.5 * length, .5 * length])
        #plt.xlim = ([-60., 25.])

        """
        if i%400==0 or i == 800 or i==1000:

            plt.savefig('visuals/' + 'frame=%.1f' %i + '.png', bbox_inches='tight', dpi=600)

        else:

            pass 
        """

        # plt.tight_layout()

        return line,

    anim = animation.FuncAnimation(fig, animate, np.shape(u)[0], blit=False)

    #plt.savefig('visuals/' + 'frame'+ '.png', bbox_inches='tight', dpi=100)

    # add the folder "visuals" to our path... more on this below
    my_path = os.path.join("visuals")

    # first, if the folder doesn't exist, make it
    if not os.path.isdir(my_path):
        os.makedirs(my_path)

    anim.save('visuals/' + filename, fps=50, extra_args=['-vcodec', 'libx264'], dpi=dpi)


def save_combomovie(u, x, length, dt, ndump, filename, dpi=100):
    # Create movie file in mp4 format. Warning: this is very slow!

    no_tex = False

    try:
        plt.rc('text', usetex=True)

    except RuntimeError:
        plt.rcParams["font.family"] = "serif"
        no_tex = True

    fig = plt.figure()

    umin = 1.05 * np.amin(u)
    umax = 1.05 * np.amax(u)

    ax = plt.axes(xlim=(-0.5 * length, 0.5 * length), ylim=(umin, umax))

    # create insert axes
    v = np.absolute(fft(u, axis=1)) ** 2

    N = np.shape(x)[0]

    k = fftshift(2. * np.pi * N * fftfreq(N) / length)

    kmin = np.amin(k)
    kmax = np.amax(k)

    vmin = 1.05 * np.amin(v)
    vmax = 1.05 * np.amax(v)

    v = fftshift(v, axes=1)

    ins = ax.inset_axes([0.69, 0.685, 0.3, 0.3], xlim=(kmin, kmax), ylim=(vmin, vmax))

    # use cubic spline interpolation to smooth the state data

    # note that spline interpolation is necessary to combat the Runge phenomenon since we have a uniform grid! Naive
    # use of barycentric interpolation is not an option.

    # add endpoint
    x_end = np.append(x, 0.5 * length)

    [M, N] = np.shape(u)

    u_end = np.zeros([M, N + 1], dtype=float)

    u_end[:, 0:N] = u

    u_end[:, -1] = np.copy(u[:, 0])  # enforce periodicity

    # create the interpolating spline polynomial using scipy interpolation package

    poly = CubicSpline(x_end, u_end, axis=1, bc_type='periodic')

    # now we can actually do the spatial upsampling

    NN = 600  # number of points to evaluate interpolant at

    xx = np.linspace(-0.5 * length, 0.5 * length, NN, endpoint=True)

    uu = poly(xx)

    x = xx

    u = uu

    # now make smaller insert graph plotting data

    poly = CubicSpline(k, v, axis=1)  # ignore periodic flag here, it's not really worth the extra effort

    NN = 3600  # number of points to evaluate interpolant at

    kk = np.linspace(kmin, kmax, NN, endpoint=True)

    vv = poly(kk)

    k = kk

    v = vv

    ax.grid(True)
    ax.set_xlabel('$x$', fontsize=22)

    ins.grid(False)
    ins.set_xlabel('$k$', fontsize=11)

    line, = ax.plot([], [], linewidth=2, color='xkcd:ocean green')
    iline, = ins.plot([], [], linewidth=1., color='xkcd:dark orange')

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(u, ax)
    timer.start()

    # Animation function.
    def animate(i):
        line.set_data(x, u[i, :])
        iline.set_data(k, v[i, :])

        tplot = i * dt * ndump

        if no_tex:
            ylabel_str = 'u(x,t=%.2f)' % tplot

        else:
            ylabel_str = r'$u(x,t=%.2f)$' % tplot


        ax.set_ylabel(ylabel_str, fontsize=22)

        # TODO: get this looking better when tex doesn't want to work!
        in_ylabel_str = '$|\widehat{u}|^2(k,t=%.2f)$' % tplot
        ins.set_ylabel(in_ylabel_str, fontsize=11)

        plt.tight_layout()

        return line,

    anim = animation.FuncAnimation(fig, animate, np.shape(u)[0], blit=False)

    # add the folder "visuals" to our path... more on this below
    my_path = os.path.join("visuals")

    # first, if the folder doesn't exist, make it
    if not os.path.isdir(my_path):
        os.makedirs(my_path)

    anim.save('visuals/' + filename, fps=200, extra_args=['-vcodec', 'libx264'], dpi=dpi)
