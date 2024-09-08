import os

from typing import ContextManager, Optional
from alive_progress import alive_bar

def spinner(title: Optional[str] = None) -> ContextManager:
    r"""Displays a nice little progress bar to indicate that certain tasks are running.

    The code is taken right from the reference, with no real input from the *joe* development team.

    Parameters
    ----------
        title : str, optional

    References
    ----------
    .. [1] <https://stackoverflow.com/questions/44851940/python-cli-progress-bar-spinner-without-iteration>
    """
    return alive_bar(monitor=None, stats=None, title=title, force_tty=True)

import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import matplotlib.animation as animation
os.environ["XDG_SESSION_TYPE"] = "xcb" # prevents a warning from being thrown up for certain linux users, thanks to
# https://stackoverflow.com/questions/69828508/warning-ignoring-xdg-session-type-wayland-on-gnome-use-qt-qpa-platform-wayland
from scipy.interpolate import CubicSpline
from numpy.fft import fft, fftfreq, fftshift


def hov_plot(x, t, u, fieldname, umin=None, umax=None, dpi=100, show_figure=True, save_figure=False, picname="",
             cmap=cmo.haline, usetex=True):
    r"""Create a Hovmoeller plot (filled contour plot in space-time) of the space-time field u.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        x : ndarray
            Spatial sample locations of the field (horizontal axis).
        t : ndarray
            Temporal sample locations of the field (vertical axis).
        u : ndarray
            Samples values of the field (axis 0 is temporal variation, axis 1 is spatial variation).
        umin : float, optional.
            Minimum field value to include, used to construct colorbar lower limit. Default: None.
        umax : float, optional.
            Maximum field value to include, used to construct colorbar upper limit. Default: None.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        cmap : str, optional
            Name of the colormap to use. For a list of great colormaps see https://matplotlib.org/cmocean/.
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
        picname : str, optional
            Name of the .png file to save the figure to. Default: "".
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    if umin is None:
        umin = np.amin(u)
    if umax is None:
        umax = np.amax(u)

    levels = np.linspace(umin, umax, num=300)

    CF = plt.contourf(x, t, u, cmap=cmap, levels=levels)

    # axis labels
    if usetex:
        plt.xlabel(r"$x$", fontsize=22, color='k')
        plt.ylabel(r"$t$", fontsize=22, color='k')

    else:
        plt.xlabel(r"x", fontsize=22, color='k')
        plt.ylabel(r"t", fontsize=22, color='k')

    plt.tick_params(axis='x', which='both', top=False, color='k')
    plt.xticks(fontsize=16, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right=False, color='k')
    plt.yticks(fontsize=16, rotation=0, color='k')

    plt.xlim([np.amin(x), np.amax(x)])

    # make colorbar
    cbar = plt.colorbar(CF, format='%.2f')
    cbar.ax.tick_params(labelsize=16, color='k')
    plt.clim(umin, umax)

    if usetex:
        cbarlabel_str = r'$u(x,t)$'.replace('u', str(fieldname))

    else:
        cbarlabel_str = 'u(x,t)'.replace('u', str(fieldname))

    cbar.ax.set_ylabel(cbarlabel_str, fontsize=22, color='k')

    # the final piece of the colorbar defn is to change the colorbar ticks to an acceptable color.
    # This is not so easy, and relies on the thread at
    # https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    cbytick_obj = plt.getp(cbar.ax.axes, 'yticklabels')
    plt.setp(cbytick_obj, color='k')

    plt.tight_layout()

    if save_figure is True:

        # add the folder "joe_visuals" to our path... more on this below
        my_path = os.path.join("joe_visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and save the fig
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=dpi)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

def nice_plot(x, y, xlabel, ylabel, dpi=100, custom_ylim=None, show_figure=True, save_figure=False, picname="",
              linestyle='solid', color='xkcd:cerulean', usetex=True):
    r"""Renders a nice 2D line plot of y as a function of x.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        x : ndarray
            Manipulated variable/horizontal axis.
        y : ndarray
            Responding variable/vertical axis.
        xlabel : str
            Label of the horizontal axis.
        ylabel : str
            Label of the vertical axis.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        custom_ylim : tuple, optional
            Custom values for the limits on the vertical axis.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        color: str, optional
            Name of the color you want the curve to be. For a list of great colors see https://xkcd.com/color/rgb/.
            Default: 'xkcd:cerulean'.
        linestyle : str, optional
            Style of the line, see matplotlib docs for a list of acceptable values:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle .
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
            all. Default: False.
        picname : str, optional
            Name of the .png file to save the figure to. Default: "".
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    fig, ax = plt.subplots()

    plt.plot(x, y, linestyle = linestyle, color=color, linewidth='2')

    plt.xlim([np.amin(x), np.amax(x)])

    if custom_ylim == None:

        pass

    else:

        plt.ylim(custom_ylim)

    # axis labels
    plt.xlabel(xlabel, fontsize=22, color='k')
    plt.ylabel(ylabel, fontsize=22, color='k')

    plt.tick_params(axis='x', which='both', top='off', color='k')
    plt.xticks(fontsize=16, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right='off', color='k')
    plt.yticks(fontsize=16, rotation=0, color='k')

    plt.tight_layout()

    if save_figure is True:

        # add the folder "joe_visuals" to our path... more on this below
        my_path = os.path.join("joe_visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and save the fig
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=dpi)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

# plot a bunch of stuff on the same axis.
def nice_multiplot(xs, ys, xlabel, ylabel, curvelabels, linestyles, colors, linewidths, custom_ylim = None,
                   dpi=100, show_figure=True,
                   save_figure=False, picname="", usetex=True):
    r"""Renders a nice 2D line plot of a bunch of y's as a function of a bunch of x's.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        xs : list of ndarrays
               Manipulated variable/horizontal axis.
        ys : list of ndarrays
            Responding variable/vertical axis.
        xlabel : str
            Label of the horizontal axis.
        ylabel : str
            Label of the vertical axis.
        curvelabels: list of strings
            Each entry is a label for the corresponding curve, used to create the legend.
        linestyles : list of strings
            Style of the plotted lines, see matplotlib docs for a list of acceptable values:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
        colors: list of strings
            Name of the colors you want each curve to be. For a list of great colors see https://xkcd.com/color/rgb/.
        linewidths : list of floats
            Widths of the plotted lines.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        custom_ylim : tuple, optional
            Custom values for the limits on the vertical axis.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
            all. Default: False.
        picname : str, optional
            Name of the .png file to save the figure to. Default: "".
    """

    if len(xs) != len(ys) or len(xs) != len(curvelabels):

        raise ValueError('xs, ys, and curvelabels must be lists of the same length.')

    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    fig, ax = plt.subplots()

    for m in range(0, len(xs)):

        plt.plot(xs[m], ys[m], linestyle=linestyles[m], color=colors[m], linewidth=linewidths[m],
                 label=curvelabels[m])

    plt.xlim([np.amin(xs), np.amax(xs)])

    if custom_ylim == None:

        pass

    else:

        plt.ylim(custom_ylim)

    ax.legend(fontsize=14, loc='best')

    # axis labels
    plt.xlabel(xlabel, fontsize=22, color='k')
    plt.ylabel(ylabel, fontsize=22, color='k')

    plt.tick_params(axis='x', which='both', top='off', color='k')
    plt.xticks(fontsize=16, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right='off', color='k')
    plt.yticks(fontsize=16, rotation=0, color='k')

    plt.tight_layout()

    if save_figure is True:

        # add the folder "joe_visuals" to our path... more on this below
        my_path = os.path.join("joe_visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and save the fig
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=dpi)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

def nice_hist(data, xlabel, dpi=100, show_figure=True, save_figure=False, picname="",
              color='xkcd:deep pink', usetex=True):
    r"""Renders a nice histogram of the given data.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        data : ndarray
            Data to be binned and plotted.
        xlabel : str
            Label of the horizontal axis.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        color: str, optional
            Name of the color you want the curve to be. For a list of great colors see https://xkcd.com/color/rgb/.
            Default: 'xkcd:deep pink'.
        show_figure: boolean, optional
            True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
        save_figure: boolean, optional
            True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
            all. Default: False.
        picname : str, optional
            Name of the .png file to save the figure to. Default: "".
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    fig = plt.figure()
    plt.hist(data, color=color)
    fig.set_size_inches(8, 6)
    plt.tick_params(axis='x', which='both', top=False, color='k')
    plt.xticks(fontsize=20, rotation=0, color='k')
    plt.tick_params(axis='y', which='both', right=False, color='k')
    plt.yticks(fontsize=20, rotation=0, color='k')

    plt.xlabel(xlabel, fontsize=22, color='k')

    plt.ylabel(r"Number of Instances", fontsize=22, color='k')
    plt.xlim([np.amin(data), np.amax(data)])

    plt.tight_layout()

    if save_figure is True:

        # add the folder "joe_visuals" to our path... more on this below
        my_path = os.path.join("joe_visuals")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        # and save the fig
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=dpi)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

def plot_refinement_study(model, initial_state, length, T, Ns, dts, errors, method_kw='etdrk4', bc='periodic',
                        show_figure=True, save_figure=False, usetex=True, dpi=400):
    r"""Plots a refinement study (see :func:`~joe_lab.joe.do_refinement_study`.

    The actual code is a bit ugly but it gets the job done and prevents the main *joe* scripts from being too cluttered.

    Parameters
    ----------
       model: model
           Instance of the class :class:`~joe_lab.joe.model`.
       initial_state: initial_state
           Instance of the class :class:`~joe_lab.joe.initial_state`.
       length : float
           Length of the spatial domain
       T : float
           Total physical runtime of the simulation. That is, the spacetime grid covers the time interval [0,T].
       Ns : list of ints
           Number of spatial locations on which to sample the solution to our PDE.
       dts : list of floats
           Time step sizes for numerical integration of the PDE.
       errors : list of ndarrays
           Estimated errors for each choice of space-time grid.
       method_kw : str, optional
           Name of the numerical method. Currently, 'etdrk1', 'ifrk4' are available for first-order-in-time problems,
           and 'etdrk4' is available for all problems: see :class:`~joe_lab.time_stepper.timestepper`.
           Default: 'etdrk4'.
       bc : str, optional
           String specifying the boundary conditions. Must be 'periodic' or 'sponge_layer'.
       dpi : int, optional
           Dots-per-inch on the image. Default: 400.
       usetex : boolean, optional
           True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
       show_figure: boolean, optional
           True if you want the figure to appear in a pop-up window once it is rendered, False otherwise. Default: True.
       save_figure: boolean, optional
           True if you want the figure to be saved as a .png file, False if you do not want the figure to be saved at
           all. Default: False.
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False

    fig, ax = plt.subplots()

    num_Ns = np.size(Ns)

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
            absorbing_layer = True
        elif bc == 'periodic':
            absorbing_layer = False

        my_string = ('_length=%.1f_T=%.1f' % (
            length, T) + '_modelkw=' + model.model_kw + '_ICkw=' + initial_state.initial_state_kw + '_method_kw='
                     + method_kw + '_nonlinear=' + str(model.nonlinear) + '_abslayer=' + str(absorbing_layer))

        picname = 'refinement_study' + my_string + '.png'
        plt.savefig('joe_visuals/' + picname, bbox_inches='tight', dpi=dpi)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    plt.close()

# create a movie from a scalar field u(t,x) sampled at various times.

def save_movie(u, x, length, dt, fieldname, ndump, filename, fps=200, periodic=True, usetex=True,
               fieldcolor='xkcd:ocean green', dpi=100):
    r"""Save a movie of the evolution of a real scalar field as a .mp4 file.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        u : ndarray
            Sampled values of our scalar field on a given space-time grid.
        length : float
            Length of the spatial domain.
        x : ndarray
            Spatial grid of [-0.5*length, 0.5*length].
        dt : float
            Lattice spacing on the time axis.
        fieldname : str
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        ndump : int
            Only every `ndump` timesteps are stored in u.
        filename : str
             Name of the .mp4 file to save the movie as.
        periodic : boolean, optional
            True if the field physically obeys periodic boundary conditions, False otherwise.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        fieldcolor: str, optional
            Name of the color you want the curve to be. For a list of great colors see https://xkcd.com/color/rgb/.
            Default: 'xkcd:ocean green'.
        fps: int, optional
            Frames-per-second for the movie. Default: 200.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError: # catch a user error thinking they have tex when they don't
        usetex = False

    fig = plt.figure()

    umin = np.amin(u)
    umax = np.amax(u)

    if umin < 0:

        umin *= 1.05

    else:

        umin *=0.95

    if umax > 0:

        umax *= 1.05

    else:

        umax *= 0.95

    #from scipy.integrate import simps

    # coeff = simps(np.abs(u[0, :]),x)  #TODO: multiplying by a factor of 4 gives a solid result , but for BBM 6 is # the
    # correct "back of the envelope" value.

    ax = plt.axes(xlim=(np.amin(x), np.amax(x)), ylim=(umin, umax))
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

    NN = 2000  # number of points to evaluate interpolant at

    xx = np.linspace(np.amin(x), np.amax(x), NN, endpoint=True)

    uu = poly(xx)

    x = xx

    u = uu

    ax.grid('True')

    line, = ax.plot([], [], linewidth=2, color=fieldcolor, label=fieldname)
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

    if usetex:
        plt.xlabel(r"$x$", fontsize=22, color='k')
    else:
        plt.xlabel(r"x", fontsize=22, color='k')

    # Animation function.
    def animate(i):
        line.set_data(x, u[i, :])

        tplot = i * dt * ndump

        #bound_line_up.set_data(x, coeff / ((1 + tplot) ** (1. / 3.)))
        #bound_line_down.set_data(x, -coeff / ((1 + tplot) ** (1. / 3.)))

        #plt.title('$t=%.2f$' % tplot, fontsize=22)

        if usetex:
            ylabel_str = r'$u(x,t=%.2f)$'.replace('u', str(fieldname)) % tplot

        else:
            ylabel_str = 'u(x,t=%.2f)'.replace('u', str(fieldname)) % tplot

        ax.set_ylabel(ylabel_str, fontsize=22)

        plt.xlim([np.amin(x), np.amax(x)])
        #plt.xlim = ([-60., 25.])

        """
        if i%400==0 or i == 800 or i==1000:

            plt.savefig('joe_visuals/' + 'frame=%.1f' %i + '.png', bbox_inches='tight', dpi=600)

        else:

            pass 
        """

        plt.tight_layout()

        return line,

    anim = animation.FuncAnimation(fig, animate, np.shape(u)[0], blit=False)

    #plt.savefig('joe_visuals/' + 'frame'+ '.png', bbox_inches='tight', dpi=100)

    # add the folder "joe_visuals" to our path... more on this below
    my_path = os.path.join("joe_visuals")

    # first, if the folder doesn't exist, make it
    if not os.path.isdir(my_path):
        os.makedirs(my_path)

    anim.save('joe_visuals/' + filename, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'], dpi=dpi)

    plt.close()

def save_combomovie(u, x, length, dt, ndump, filename, fieldname,
                    fps=200, periodic=True, usetex=True, fieldcolor='xkcd:ocean green', speccolor='xkcd: dark magenta',
                    dpi=100):
    r"""Save a movie of the evolution of a real scalar field as a .mp4 file, and a tinier embedded movie of its
    power spectrum.

    This just wraps matplotlib code with a bunch of nice presets, labels, etc.!

    Parameters
    ----------
        u : ndarray
            Sampled values of our scalar field on a given space-time grid.
        length : float
            Length of the spatial domain.
        x : ndarray
            Spatial grid of [-0.5*length, 0.5*length].
        dt : float
            Lattice spacing on the time axis.
        fieldname : str
            Name of the PDE solution, which will also be the label on the picture's y-axis. Default: 'u'.
        ndump : int
            Only every `ndump` timesteps are stored in u.
        filename : str
            Name of the .mp4 file to save the movie as.
        periodic : boolean, optional
            True if the field physically obeys periodic boundary conditions, False otherwise.
        usetex : boolean, optional
            True if you want to render the plot labels in TeX, and False if you want no TeX. Default: True.
        fieldcolor: str, optional
            Name of the color you want the physical-space curve to be. For a list of great colors see
            https://xkcd.com/color/rgb/. Default: 'xkcd:ocean green'.
        speccolor: str, optional
            Name of the color you want the Fourier-space power spectrum curve to be. Default: 'xkcd:dark magenta'.
        fps: int, optional
            Frames-per-second for the movie. Default: 200.
        dpi : int, optional
            Dots-per-inch on the image. Default: 100.
    """
    plt.rcParams["font.family"] = "serif"

    try:
        plt.rc('text', usetex=usetex)

    except RuntimeError:  # catch a user error thinking they have tex when they don't
        usetex = False


    fig = plt.figure()

    umin = 1.05 * np.amin(u)
    umax = 1.05 * np.amax(u)

    ax = plt.axes(xlim=(np.amin(x), np.amax(x)), ylim=(umin, umax))

    # create insert axes
    v = np.absolute(fft(u, axis=1)) ** 2

    N = np.shape(x)[0]

    k = fftshift(2. * np.pi * N * fftfreq(N) / length)

    kmin = np.amin(k)
    kmax = np.amax(k)

    vmin = 0.9 * np.amin(v)
    vmax = 1.05 * np.amax(v)

    v = fftshift(v, axes=1)

    ins = ax.inset_axes([0.15, 0.64, 0.3, 0.3], xlim=(kmin, kmax), ylim=(vmin, vmax))
    # phi4 vals: [0.69, 0.685, 0.3, 0.3]

    # use cubic spline interpolation to smooth the state data

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

    # now we can actually do the spatial upsampling

    NN = 2000  # number of points to evaluate interpolant at

    xx = np.linspace(np.amin(x), np.amax(x), NN, endpoint=True)

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
    if usetex:
        ax.set_xlabel('$x$', fontsize=22)
    else:
        ax.set_xlabel('x', fontsize=22)

    ins.grid(False)
    if usetex:
        ins.set_xlabel('$k$', fontsize=11)
    else:
        ins.set_xlabel('k', fontsize=11)

    line, = ax.plot([], [], linewidth=2, color=fieldcolor)
    iline, = ins.plot([], [], linewidth=1., color=speccolor)

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(u, ax)
    timer.start()

    # Animation function.
    def animate(i):
        line.set_data(x, u[i, :])
        iline.set_data(k, v[i, :])

        tplot = i * dt * ndump

        if usetex:
            ylabel_str = r'$u(x,t=%.2f)$'.replace('u', str(fieldname)) % tplot

        else:
            ylabel_str = 'u(x,t=%.2f)'.replace('u', str(fieldname)) % tplot

        ax.set_ylabel(ylabel_str, fontsize=22)

        if usetex:
            in_ylabel_str = r'$|\widehat{u}|^2(k,t=%.2f)$'.replace('u', str(fieldname)) % tplot

        else:
            in_ylabel_str = 'Pow[u](k,t=%.2f)'.replace('u', str(fieldname)) % tplot

        ins.set_ylabel(in_ylabel_str, fontsize=11)

        plt.tight_layout()

        return line,

    anim = animation.FuncAnimation(fig, animate, np.shape(u)[0], blit=False)

    # add the folder "joe_visuals" to our path... more on this below
    my_path = os.path.join("joe_visuals")

    # first, if the folder doesn't exist, make it
    if not os.path.isdir(my_path):
        os.makedirs(my_path)

    anim.save('joe_visuals/' + filename, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=dpi)

    plt.close()