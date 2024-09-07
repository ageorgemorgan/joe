import joe_lab.joe as joe
import joe_lab.models as joe_models
import joe_lab.initial_states as joe_initial_states

from testing_utils import do_it_all

# test whether a code for a real field can run, at least for a short time, regardless of the BC
# crucially, these particular tests won't tell us if the code is accurate or not, just that the code actually converged!

my_model = joe_models.builtin_model('phi4pert', nonlinear=True)
my_initial_state = joe_initial_states.builtin_initial_state('gaussian_odd')

length, T, N, dt = 240.,4., 2**9, 1e-2
stgrid = {'length': length, 'T': T, 'N': N, 'dt': dt}

def test_periodic_bcs():
    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='periodic', ndump=20)
    converged = do_it_all(my_sim)

    # test that the code actually ran
    assert converged == True

def test_sponge_layer_bcs():
    l_endpt = -0.5 * length + 0.5 * length * 0.05
    r_endpt = l_endpt + 0.05 * length
    width = (2 ** -4) * length / 100.
    sponge_params = {'l_endpt': l_endpt, 'r_endpt': r_endpt,
                     'width': width, 'expdamp_freq': 1e3,
                     'damping_amplitude': 10.,
                     'spongeless_frac': 0.5}  # this is the fraction of the middle of the spatial domain to keep in the plots

    my_sim = joe.simulation(stgrid, my_model, my_initial_state, bc='sponge_layer', sponge_params=sponge_params,
                            ndump=20)

    converged = do_it_all(my_sim)

    # test that the code actually ran
    assert converged == True
