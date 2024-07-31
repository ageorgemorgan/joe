import pickle

import os

import numpy as np

import time

from alive_progress import alive_bar

from numpy.fft import fft, ifft, fftfreq

from scipy import sparse

from scipy.sparse import linalg, diags

from absorbing_layer import damping_coeff_lt, rayleigh_damping


# The intention here is to make the code independent of the particular
# PDE we're considering insofar as is possible.


# First, a function for computing all of the Greeks ("weights" for exponential quadrature).
# We do this by Pythonizing the code from Kassam and Trefethen 2005 (do Cauchy integrals).

def get_greeks_first_order(N, dt, z):
    M = 2 ** 5

    theta = np.linspace(0., 2. * np.pi, num=M, endpoint=False)

    rad = 1.  # radius of contour surrounding dt*z over which we integrate

    z0 = dt * np.tile(z, (M, 1)) + rad * np.tile(np.exp(1j * theta), (N, 1)).T

    Q = dt * np.real(np.mean((np.exp(0.5 * z0) - 1.) / z0, 0))  # note how we take mean over a certain axis

    f1 = dt * np.real(np.mean((-4. - z0 + np.exp(z0) * (4. - 3. * z0 + z0 ** 2)) / (z0 ** 3), 0))  # again note axis
    # argument of np.mean

    f2 = dt * np.real(np.mean((2. + z0 + np.exp(z0) * (-2. + z0)) / (z0 ** 3), 0))

    f3 = dt * np.real(np.mean((-4. - 3. * z0 - z0 ** 2 + np.exp(z0) * (4. - z0)) / (z0 ** 3), 0))

    out = [Q, f1, f2, f3]

    return out


def get_greeks_second_order(length, N, dt, A):
    M = 2 ** 5
    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    # radius of contour = largest eigenvalue of linear part with a bit of wiggle room
    max_freq = np.pi * N / length
    rad = 1.2 * dt * np.sqrt(max_freq ** 2 + 2.)
    r = rad * np.exp(1j * theta)

    id_matrix = sparse.eye(2 * N, dtype=float)

    Q = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f1 = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f2 = 1j * np.zeros([2 * N, 2 * N], dtype=float)
    f3 = 1j * np.zeros([2 * N, 2 * N], dtype=float)

    for j in np.arange(0, M):
        z = r[j]

        B = id_matrix.multiply(z) - A.multiply(dt)

        B = sparse.csc_matrix(B)

        zIA = sparse.linalg.inv(B)

        Q += dt * zIA * (np.exp(0.5 * z) - 1.)
        f1 += dt * zIA * ((-4. - z + np.exp(z) * (4. - 3. * z + z ** 2)) / (z ** 2))
        f2 += dt * zIA * ((2. + z + np.exp(z) * (-2. + z)) / (z ** 2))
        f3 += dt * zIA * ((-4. - 3. * z - z ** 2 + np.exp(z) * (4. - z)) / (z ** 2))

    Q = sparse.csc_matrix(np.real(Q / M))
    f1 = sparse.csc_matrix(np.real(f1 / M))
    f2 = sparse.csc_matrix(np.real(f2 / M))
    f3 = sparse.csc_matrix(np.real(f3 / M))

    out = [Q, f1, f2, f3]

    return out


def get_Q1(N, dt, z):
    M = 2 ** 5

    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    rad = 1.  # radius of contour about dt*z about which we integrate

    z0 = dt * np.tile(z, (M, 1)) + rad * np.tile(np.exp(1j * theta), (N, 1)).T

    out = dt * np.real(np.mean((np.exp(z0) - 1.) / z0, 0))  # note how we take mean over a certain axis

    return out


def get_R2(N, dt, z):
    # """
    M = 2 ** 5

    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    rad = 1.  # radius of contour about dt*z about which we integrate

    z0 = dt * np.tile(z, (M, 1)) + rad * np.tile(np.exp(1j * theta), (N, 1)).T

    out = dt * np.real(np.mean((np.exp(z0) - 1. - z0) / (z0 ** 2), 0))  # note how we take mean over a certain axis
    # """

    return np.real(out)  # just kill any error that arose


def do_etdrk1_step(V, propagator, forcing, Q1):
    # remark on notation: Q1 = dt*phi1(dt*A)

    out = propagator * V + Q1 * forcing(V)

    return out


def do_etdrk2_step(V, propagator, forcing, Q1, R2):
    a = do_etdrk1_step(V, propagator, forcing, Q1)

    out = a + R2 * (forcing(a) - forcing(V))

    return out


# code for a single ETDRK4 step, for a first-order-in-time PDE
def do_etdrk4_step(V, propagator, propagator2, forcing, greeks):
    Q = greeks['Q']
    f1 = greeks['f1']
    f2 = greeks['f2']
    f3 = greeks['f3']

    fV = forcing(V)

    Vhalf = propagator2 * V

    a = Vhalf + Q * fV

    fa = forcing(a)

    b = Vhalf + Q * fa

    fb = forcing(b)

    c = propagator2 * a + Q * (2. * fb - fV)

    fc = forcing(c)

    # now assemble the guess at the new step
    out = propagator * V + f1 * fV + 2. * f2 * (fa + fb) + f3 * fc

    return out


# code for a single ETDRK4 step, if the eqn is second order in time (basically the above code, but sparsified)
def do_etdrk4_step_second_order(V, propagator, propagator2, forcing, greeks):
    Q = greeks['Q']
    f1 = greeks['f1']
    f2 = greeks['f2']
    f3 = greeks['f3']

    N = int(0.5 * np.size(V))

    fV = forcing(V)

    Vhalf = propagator2 @ V  # note: @ takes advantage of sparsity.

    a = Vhalf + np.asarray(Q @ fV)

    a = np.reshape(a, (2 * N,))

    fa = forcing(a)

    b = Vhalf + np.asarray(Q @ fa)

    b = np.reshape(b, (2 * N,))

    fb = forcing(b)

    c = np.asarray(propagator2 @ a + Q @ (2. * fb - fV))

    c = np.reshape(c, (2 * N,))

    fc = forcing(c)

    # now assemble the guess at the new step. This is the temporal bottleneck of the time step (probably like 70% of step time)
    out = np.asarray(propagator @ V + f1 @ fV + 2. * f2 @ (fa + fb) + f3 @ fc)

    out = np.reshape(out, (2 * N,))

    return out


# code for a single integrating factor Runge-Kutta fourth-order step
def do_ifrk4_step(V, propagator, propagator2, forcing, dt):
    a = dt * forcing(V)

    b = dt * forcing(propagator2 * (V + 0.5 * a))

    c = dt * forcing(propagator2 * V + 0.5 * b)

    d = dt * forcing(propagator * V + propagator2 * c)

    out = propagator * V + (1. / 6.) * (propagator * a + 2. * propagator2 * (b + c) + d)

    return out


def assemble_damping_mat(N, length, x, dt):
    # By "damping mat", we mean the matrix to be inverted at each time step in the damping stage.
    # Currently only backward Euler inversion is implemented.
    # TODO: try out Crank-Nicolson as well, perform cost v. accuracy analysis?
    k = 2 * np.pi * N * fftfreq(N) / length

    # Deal w/ the damping mat as a scipy.sparse LinearOperator to avoid matrix mults!
    # This is an issue here bcz dealing with the matrix of the Fourier transform is a pain

    def mv(v):
        # NOTE: v is given ON THE FOURIER SIDE!!!!

        mv_out = v - dt * (-1j * k) * fft(damping_coeff_lt(x, length) * np.real(ifft((-1j * k) * v)))

        return mv_out

    out = linalg.LinearOperator(shape=(N, N), matvec=mv)

    # TODO: apparently LinearOperators can't be pickled, but sparse matrices can? See if there is some other decent
    # way of saving LinearOperators

    return out


def do_diffusion_step(q, B):
    B_LHS = B

    RHS = q

    out, info = linalg.cg(B_LHS, RHS)  # have to have "info" here otherwise the code throws a fit

    return out


class timestepper:
    def __init__(self, method_kw, sim, scale=1.):
        self.method_kw = method_kw
        self.sim = sim
        self.scale = scale
        self.t_ord = sim.model.t_ord
        self.aux = None
        dt_new = scale*sim.dt
        self.auxfilename = 'timestepper_auxfile_method_kw=' + self.method_kw + '_length=%.1f_T=%.1f_N=%.1f_dt=%.6f' % (
            sim.length, sim.T, sim.N,
            dt_new) + '_modelkw=' + sim.model_kw + '.pkl'  # need to save as pkl since we store aux as a dict

    def get_aux(self):

        sim = self.sim

        t_ord = self.t_ord

        length = sim.length

        N = sim.N

        dt = self.scale*sim.dt

        k = 2. * np.pi * N * fftfreq(N) / length

        A = sim.model.get_symbol(k)

        if t_ord == 1:

            propagator = np.exp(dt * A)

        elif t_ord == 2:

            A = sparse.diags([A, np.ones(N, dtype=float)], [-N, N], shape=[2 * N, 2 * N]).tocsc()

            propagator = linalg.expm(A.multiply(dt))

        if self.method_kw == 'etdrk1':
            Q1 = get_Q1(N, dt, A)

            aux = dict([('Q1', Q1), ('propagator', propagator)])

        if self.method_kw == 'etdrk4':

            if t_ord == 1:

                [Q, f1, f2, f3] = get_greeks_first_order(N, dt, A)

                propagator2 = np.exp(0.5 * dt * A)

            elif t_ord == 2:

                [Q, f1, f2, f3] = get_greeks_second_order(length, N, dt, A)

                propagator2 = linalg.expm(A.multiply(0.5*dt))

            aux = dict([('Q', Q), ('f1', f1), ('f2', f2), ('f3', f3), ('propagator', propagator),
                        ('propagator2', propagator2)])

        if self.method_kw == 'ifrk4':
            propagator2 = np.exp(0.5 * dt * A)

            aux = dict([('propagator', propagator), ('propagator2', propagator2)])

        self.aux = aux

    def save_aux(self):

        # add the folder "timestepper_aux" to our path... more on this below
        my_path = os.path.join("timestepper_aux")

        # first, if the folder doesn't exist, make it
        if not os.path.isdir(my_path):
            os.makedirs(my_path)

        with open('timestepper_aux/'+self.auxfilename, 'wb') as outp:
            pickle.dump(self.aux, outp, pickle.HIGHEST_PROTOCOL)

    def load_aux(self):

        with open('timestepper_aux/'+self.auxfilename, 'rb') as inp:
            self.aux = pickle.load(inp)

    def do_time_step(self, V, forcing):

        t_ord = self.t_ord

        aux = self.aux

        propagator = aux['propagator']

        if self.method_kw == 'etdrk1':
            Q1 = aux['Q1']

            out = do_etdrk1_step(V, propagator, forcing, Q1)

        if self.method_kw == 'etdrk4':
            Q = aux['Q']

            f1 = aux['f1']

            f2 = aux['f2']

            f3 = aux['f3']

            greeks = dict([('Q', Q), ('f1', f1), ('f2', f2), ('f3', f3)])

            propagator = aux['propagator']

            propagator2 = aux['propagator2']

            if t_ord == 1:

                out = do_etdrk4_step(V, propagator, propagator2, forcing, greeks)

            if t_ord == 2:

                out = do_etdrk4_step_second_order(V, propagator, propagator2, forcing, greeks)

        if self.method_kw == 'ifrk4':
            propagator = aux['propagator']

            propagator2 = aux['propagator2']

            out = do_ifrk4_step(V, propagator, propagator2, forcing, self.scale*self.sim.dt)

        return out


def do_time_stepping(sim, method_kw='etdrk4', splitting_method_kw='naive'):
    length = sim.length

    T = sim.T

    N = sim.N

    dt = sim.dt

    model = sim.model

    t_ord = model.t_ord

    initial_state = sim.initial_state

    nonlinear = sim.nonlinear

    absorbing_layer = sim.absorbing_layer

    ndump = sim.ndump

    nsteps = int(T / dt)

    x = sim.x  # the endpoint = False flag is critical!

    k = 2. * np.pi * N * fftfreq(N) / length

    # determine the time-step scale factor "a" for splitting
    if splitting_method_kw == 'strang':

        scale = 0.5

    else:

        scale = 1.

    my_timestepper = timestepper(method_kw, sim, scale=scale)

    # preprocessing stage: assemble the aux quantities needed for time-stepping, and the forcing function

    # create forcing term

    def forcing(V):

        out = model.get_fourier_forcing(V, k, x, nonlinear)

        # if we're second-order in time and using a sponge layer, damping can be realized simply
        # by modifying the forcing term ie. damping can be dealt with explicitly!
        if t_ord == 2 and absorbing_layer:

            out += rayleigh_damping(V, x, length,  delta=0.25 * length)

        return out

    # obtain the aux quantities. Thanks to all the hard work we did when defining the timestepper class, the code here
    # is brief and (IMO) elegant.

    # first check if we've already computed aux on the required space-time grid
    try:

        my_timestepper.load_aux()

    # if the auxfile is not found, compute aux here.
    except:

        my_timestepper.get_aux()
        my_timestepper.save_aux()

    # now assemble the stuff needed for damping, if needed
    if absorbing_layer and t_ord == 1:
        damping_mat = assemble_damping_mat(N, length, x, dt)
    else:
        pass

    Uinit = initial_state

    if t_ord == 2:

        try:

            v1 = fft(Uinit[0, :])
            v2 = fft(Uinit[1, :])

        except: # if no initial speed is provided in second order case, default to assuming it's zero.

            v1 = fft(Uinit)
            v2 = np.zeros_like(Uinit, dtype=float)

        V = np.concatenate((v1, v2))

        # make data storage array
        Udata = np.zeros([2, 1 + int(nsteps / ndump), N], dtype=float)
        Udata[:, 0, :] = Uinit

    elif t_ord == 1:

        V = fft(Uinit)

        # make data storage array
        Udata = np.zeros([1 + int(nsteps / ndump), N], dtype=float)
        Udata[0, :] = Uinit

    cnt = 0.  # counter

    for n in np.arange(1, nsteps + 1):

        Va = my_timestepper.do_time_step(V, forcing)

        if absorbing_layer and t_ord == 1:

            if splitting_method_kw == 'naive':

                Vb = do_diffusion_step(Va, damping_mat)

                V = Vb

            elif splitting_method_kw == 'strang':

                Vb = do_diffusion_step(Va, damping_mat)

                Vc = my_timestepper.do_time_step(Vb, forcing)

                V = Vc

            if cnt % 500 == 0:

                # TODO: be able to toggle "harsh" exponential damping!
                pass

                U = np.real(ifft(V))

                U *= 1. - 1. * damping_coeff_lt(-x, length)

                V = fft(U)

        else:

            V = Va

        cnt += 1

        # data storage step
        if cnt % ndump == 0:

            if t_ord == 2:

                Udata[0, int(n / ndump), :] = np.real(ifft(V[0:N]))
                Udata[1, int(n / ndump), :] = np.real(ifft(V[N:]))

            else:

                Udata[int(n / ndump), :] = np.real(ifft(V))

        else:

            pass

    return Udata
