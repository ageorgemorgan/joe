import pickle
import os

import time

import numpy as np
import matplotlib.pyplot as plt

from joe_main_lib import simulation


def do_refinement_study(model, initial_state, length, T, Ns, dts, bc='periodic', show_figure=True, save_figure=False,
                        fit_min=3, fit_max=7):
    Ns = Ns.astype(int)
    num_Ns = np.size(Ns)
    num_dts = np.size(dts)

    # initialize outputs

    errors = np.zeros([num_Ns, num_dts], dtype=float)

    cnt = 0

    start = time.time()
    print('Running accuracy tests...')
    for k in np.arange(0, num_Ns):

        N = Ns[k]

        # do simulation at the worst order (largest time step) first
        rough_sim = simulation(length, T, N, dts[0], model=model, initial_state=initial_state, bc=bc)

        rough_filename = rough_sim.filename

        try:
            # load the pkl file
            with open('sim_archive/' + rough_filename, 'rb') as inp:
                rough_sim = pickle.load(inp)

        except:

            rough_sim.run_sim(print_runtime=False)

            rough_sim.save()

        x = rough_sim.x  # same for both rough and fine

        for dt in dts:

            fine_sim = simulation(length, T, N, 0.5 * dt, model=model, initial_state=initial_state, bc=bc)

            fine_filename = fine_sim.filename

            try:
                # load the pkl file
                with open('sim_archive/' + fine_filename, 'rb') as inp:
                    fine_sim = pickle.load(inp)

            except:

                fine_sim.run_sim(print_runtime=False)

                fine_sim.save()

            rough_Udata = rough_sim.Udata  # [:, int(N/4):int(3*N/4)]

            fine_Udata = fine_sim.Udata  # [:, int(N/4):int(3*N/4)]

            # use fine sim and rough sim at last time step to get Richardson error estimate

            ord = 4.

            errors[k, cnt] = (1. / (2 ** (ord - 1))) * np.amax(np.abs(rough_Udata[-1, :] - fine_Udata[-1, :]))

            rough_sim = fine_sim  # redefine for efficiency... only works bcz we refine dt in powers of 1/2

            cnt += 1

        cnt = 0  # reinit the counter

    end = time.time()
    runtime = end - start
    print('Runtime for accuracy tests = %.4f' % runtime + ' s')

    # now we produce a plot of the errors
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

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

        plt.loglog(dts, errors[m, :], label=r'$N = z$'.replace('z', str(Ns[m])))
        # ^ an awesome trick from
        # https://stackoverflow.com/questions/33786332/matplotlib-using-variables-in-latex-expressions
        # was used to get the labels working as above

    ax.legend(fontsize=16)

    plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
    plt.ylabel(r"Absolute Error", fontsize=26, color='k')

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
            absorbing_layer = True
        elif bc == 'periodic':
            absorbing_layer = False

        my_string = '_length=%.1f_T=%.1f' % (
        length, T) + '_modelkw=' + model.model_kw + '_ICkw=' + initial_state.initial_state_kw + '_nonlinear=' + str(
            model.nonlinear) + '_abslayer=' + str(absorbing_layer)

        picname = 'refinement_study' + my_string + '.png'
        plt.savefig('visuals/' + picname, bbox_inches='tight', dpi=400)

    else:

        pass

    if show_figure is True:

        plt.show()

    else:

        pass

    # estimate the slope of particular error curves if you want. Needs a bit of by-hand tweaking (controlled by the
    # inputs fit_min, fit_max) bcz for small enough dt we can get level-off or rounding error domination in the error
    # curve, destroying the linear trend after a certain threshold

    params = np.polyfit(np.log10(dts[fit_min:fit_max+1]), np.log10(errors[-1, fit_min:fit_max+1]), 1)
    slope = params[0]
    print('Estimated Slope of Error Line at N = %i' % Ns[-1] + ' is slope = %.3f' % slope)

    return None
