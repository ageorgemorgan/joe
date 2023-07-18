import pickle

import numpy as np
import numpy.linalg

import matplotlib.pyplot as plt

from simulation_lib import simulation

import time


# here, we perform a nonlinear refinement study (in dt) to assess the accuracy of our nonlinear solver. Of course,
# refinement in N is a bit suspect since we use a spectral discretization in space!

# start to get the simulations ready

T = 1.  # time to stop simulation at

length = 400.

nmin = 7

nmax = 10

# prescribe the array of dt's we seek to assess
dts = np.flip(np.logspace(-nmax, -nmin, num=nmax-nmin+1, base=2.))
num_dts = np.size(dts)

# prescribe the array of N's we seek to assess
Ns = np.array([2**8, 2**9, 2**10, 2**11])
Ns = Ns.astype(int)
num_Ns = np.size(Ns)

# set what initial condition we want to deal with

model_kw = 'bbm'

ICkw = 'gaussian_even'

nonlinear = True

absorbing_layer = False

# initialize outputs

errors = np.zeros([num_Ns, num_dts], dtype=float)

cnt = 0

start = time.time()

for k in np.arange(0, num_Ns):

    N = Ns[k]

    # do simulation at the worst order (largest time step) first

    rough_sim = simulation(length, T, N, dts[0], model_kw=model_kw, initial_state_kw=ICkw, nonlinear=nonlinear, absorbing_layer=absorbing_layer)

    rough_filename = rough_sim.filename

    try:
        # load the pkl file
        with open(rough_filename, 'rb') as inp:
            rough_sim = pickle.load(inp)

    except:

        rough_sim.run_sim()

        rough_sim.save()

    x = rough_sim.x  # same for both rough and fine

    for dt in dts:

        fine_sim = simulation(length, T, N, 0.5*dt,  model_kw=model_kw, initial_state_kw=ICkw, nonlinear=nonlinear, absorbing_layer=absorbing_layer)

        fine_filename = fine_sim.filename

        try:
            # load the pkl file and try plotting again
            with open(fine_filename, 'rb') as inp:
                fine_sim = pickle.load(inp)

        except:

            fine_sim.run_sim()

            fine_sim.save()

        rough_Udata = rough_sim.Udata

        fine_Udata = fine_sim.Udata

        # use fine sim and rough sim at last time step to get Richardson error estimate

        errors[k, cnt] = (1./15.)*np.amax(np.abs(rough_Udata[-1, :] - fine_Udata[-1, :]))

        rough_sim = fine_sim  # redefine for efficiency... only works bcz we refine dt in powers of 1/2

        cnt += 1

        print(cnt)

    cnt = 0  # reinit the counter

end = time.time()
runtime = end-start
print('Runtime for accuracy tests = ', runtime, 's')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax = plt.subplots()

dts = 0.5*dts

plt.loglog(dts, errors[0, :], 'o', color='xkcd:deep green', markersize='8', label=r"$N=256$")
plt.loglog(dts, errors[0, :],  color='xkcd:deep green', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[1, :], '*', color='xkcd:raspberry', markersize='8', label=r"$N=512$")
plt.loglog(dts, errors[1, :],  color='xkcd:raspberry', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[2, :], '^', color='xkcd:goldenrod', markersize='8', label=r"$N=1024$")
plt.loglog(dts, errors[2, :],  color='xkcd:goldenrod', linewidth='2', linestyle='solid')
plt.loglog(dts, errors[3, :], 'v', color='xkcd:slate', markersize='8', label=r"$N=2048$")
plt.loglog(dts, errors[3, :],  color='xkcd:slate', linewidth='2', linestyle='solid')

ax.legend(fontsize=16)

plt.xlabel(r"$\Delta t$", fontsize=26, color='k')
plt.ylabel(r"Errors", fontsize=26, color='k')

plt.tick_params(axis='x', which='both', top='off', color='k')
plt.xticks(fontsize=16, rotation=0, color='k')
plt.tick_params(axis='y', which='both', right='off', color='k')
plt.yticks(fontsize=16, rotation=0, color='k')

plt.tight_layout()

plt.savefig('nonlinear_accuracy_test_gaussian_even', bbox_inches='tight', dpi=800)

plt.show()

#"""

# estimate the slope of particular error curves if you want. Needs a bit of by-hand tweaking bcz for small enough dt
# we can get level-off or rounding error domination in the error curve, destroying the linear trend after a certain
# threshold

params = np.polyfit(np.log10(dts), np.log10(errors[-1,:]), 1)
slope = params[0]

print('Estimated slope at N = 2048 is slope = ', slope)
#"""
