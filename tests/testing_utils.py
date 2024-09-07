def do_it_all(sim):
    r"""Helper function for testing. Runs most of the common methods for a simulation instance
     :class:`~joe_lab.joe.simulation`."""
    converged = False

    try:
        sim.plot_initial_condition(usetex=False, show_figure=False, save_figure=False)
        sim.load_or_run(method_kw='etdrk4', print_runtime=False, save=False, verbose=False)
        sim.hov_plot(cmap='cmo.haline', fieldname='u', show_figure=False, save_figure=False, usetex=True)
        sim.get_fm()
        sim.get_sm()

        converged = True

    except BaseException:
        pass

    return converged