import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

from testing_utils import do_it_all

# test whether a code for a real field can run, at least for a short time, regardless of the BC
# crucially, these particular tests won't tell us if the code is accurate or not, just that the code actually converged!

my_model = joe_models.builtin_model('kdv', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('kdv_soliton')

def test_periodic_bcs():
    length, T, N, dt = 100., 2., 2 ** 10, 1e-2
    stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}
    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)

    converged = do_it_all(my_sim)

    try:
        my_sim.save_movie(dpi=20, fps=100, usetex=False)
    except BaseException:
        converged = False

    # test that the code actually ran
    assert converged == True

def test_sponge_layer_bcs():
    length, T, N, dt = 400., 2., 2 ** 10, 1e-2
    stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

    l_endpt = -length * 0.5 + 0.5 * length * 0.1
    r_endpt = l_endpt + 0.01 * length
    width = (2 ** -6) * length / 100.
    sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                     'width': width, 'expdamp_freq': 1000,
                     'damping_amplitude': 10.,
                     'splitting_method_kw': 'naive',
                     'spongeless_frac': .5}

    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params,
                            ndump=20)

    converged = do_it_all(my_sim)

    # test that the code actually ran
    assert converged == True