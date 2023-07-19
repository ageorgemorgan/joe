import numpy as np

import time

import sys

from numpy.fft import fft, ifft, fftfreq

from scipy import sparse
from scipy.sparse import linalg, diags

from models import fourier_forcing, get_spatial_operator

from absorbing_layer import damping_coeff


# The intention with this script is to independent of the particular
# PDE we're considering insofar as is possible.


# First, a function for computing all of the Greeks ("weights" for exponential quadrature).
# We do this by Pythonizing the code from Kassam and Trefethen 2005 (do Cauchy integrals).

def get_greeks_first_order(N, dt, z, model_kw):
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

    # for efficiency, save Greeks on a particular grid.
    # filename = 'greeks_' + model_kw + '_N=%.1f_dt=%.6f' % (N, dt) + '.npz'
    # np.savez(filename, Q=Q, f1=f1, f2=f2, f3=f3)

    # TODO: make sure this saves in a separate folder to avoid cluttering the project! Learn how to do this.

    return out


def get_greeks_second_order(length, N, dt, A, model_kw):
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

    Q = np.real(Q / M)
    f1 = np.real(f1 / M)
    f2 = np.real(f2 / M)
    f3 = np.real(f3 / M)

    out = [Q, f1, f2, f3]

    # for efficiency, save Greeks on a particular grid.
    filename = 'greeks_' + model_kw + '_length=%.1f_N=%.1f_dt=%.6f' % (length, N, dt) + '.npz'
    np.savez(filename, Q=Q, f1=f1, f2=f2, f3=f3)

    # TODO: make sure this saves in a separate folder to avoid cluttering the project! Learn how to do this.

    return out


def get_Q1(N, dt, z, model_kw):
    M = 2 ** 5

    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    rad = 1.  # radius of contour about dt*z about which we integrate

    z0 = dt * np.tile(z, (M, 1)) + rad * np.tile(np.exp(1j * theta), (N, 1)).T

    # print(z0[np.abs(z0) <1e-3]) # check to see if any of the sample points generated are too small.

    out = dt * np.real(np.mean((np.exp(z0) - 1.) / z0, 0))  # note how we take mean over a certain axis

    # print(out[int(0.5*N)-5:int(0.5*N)+6])

    # TODO: implement saving of this stuff

    return out


def get_R2(N, dt, z, model_kw):
    # """
    M = 2 ** 5

    theta = np.linspace(0, 2. * np.pi, num=M, endpoint=False)

    rad = 1.  # radius of contour about dt*z about which we integrate

    z0 = dt * np.tile(z, (M, 1)) + rad * np.tile(np.exp(1j * theta), (N, 1)).T

    # print(z0[np.abs(z0) < 1e-3]) # check to see if any generated z's are too close to zero

    out = dt * np.real(np.mean((np.exp(z0) - 1. - z0) / (z0 ** 2), 0))  # note how we take mean over a certain axis
    # """

    """
    # a direct computation to show that the formula itself is what's weird!
    out = np.zeros_like(z)

    out[0] = 0.5

    zh = dt*z[1:]  # if we remove the dt, we get a better (but not ideal) shape. If we remove and mult out by dt again, we're 100% good!

    out[1:] = (np.exp(zh) - 1.-zh) / ((zh)**2)

    out *= dt

    #print(np.amax(np.abs(out)))
    #print(out)
    """

    # print(out[int(0.5*N)-5:int(0.5*N)+6])

    # TODO: implement saving of this stuff

    return np.real(out)  # just kill any error that arose


def do_etdrk1_step(V, propagator, forcing, Q1):
    # remark on notation: Q1 = dt*phi1(dt*A)

    out = propagator * V + Q1 * forcing(V)

    return out


def do_etdrk2_step(V, propagator, forcing, Q1, R2):
    a = do_etdrk1_step(V, propagator, forcing, Q1)

    out = a + R2 * (forcing(a) - forcing(V))

    return out


# code for a single ETDRK4 step
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
    # TODO: try out Crank-Nicolson as well, cost v. accuracy analysis?
    k = 2 * np.pi * N * fftfreq(N) / length

    # Deal w/ the damping mat as a scipy.sparse LinearOperator to avoid matrix mults!
    # This is an issue here bcz dealing with the matrix of the Fourier transform is a pain

    def mv(v):
        # NOTE: v is given ON THE FOURIER SIDE!!!!

        mv_out = v - dt * (-1j * k) * fft(damping_coeff(x, length) * np.real(ifft((-1j * k) * v)))

        return mv_out

    out = linalg.LinearOperator(shape=(N, N), matvec=mv)

    # TODO: apparently LinearOperators can't be pickled, but sparse matrices can? See if there is some other decent
    # way of saving LinearOperators

    return out


def do_diffusion_step(q, dt, N, B):
    B_LHS = B

    RHS = q

    out, info = linalg.cg(B_LHS, RHS)  # have to have "info" here otherwise the code throws a fit

    return out


def do_time_stepping(sim):
    length = sim.length

    T = sim.T

    N = sim.N

    dt = sim.dt

    model_kw = sim.model_kw

    initial_state = sim.initial_state

    nonlinear = sim.nonlinear

    absorbing_layer = sim.absorbing_layer

    ndump = sim.ndump

    nsteps = int(T / dt)

    x = sim.x  # the endpoint = False flag is critical!

    # preprocessing stage: assemble the spatial operator,
    # the Greeks needed for exponential time-stepping, and
    # the propagators

    A = get_spatial_operator(length, N, model_kw)

    # create forcing term

    def forcing(V):

        return fourier_forcing(V, x, length, model_kw, nonlinear=nonlinear)

    # obtain the Greeks.
    # first check if we've already computed the Greeks on the required grid

    try:

        if not absorbing_layer:

            # filename = 'Q1_' + model_kw + '_length=%.1f_N=%.1f_dt=%.6f' % (length, N, dt) + '.npz'
            filename = 'greeks_' + model_kw + '_length=%.1f_N=%.1f_dt=%.6f' % (length, N, dt) + '.npz'

        elif absorbing_layer:  # be cautious that the right Greeks are used, depending on if we need splitting or not!

            # filename = 'Q1_' + model_kw + '_length=%.1f_N=%.1f_dt=%.6f' % (length, N, 0.5*dt) + '.npz'
            filename = 'greeks_' + model_kw + '_length=%.1f_N=%.1f_dt=%.6f' % (length, N, 0.5 * dt) + '.npz'

        greeks_file = np.load(filename)  # a dictionary-like "npzfile" object

        # """
        Q = greeks_file['Q']
        f1 = greeks_file['f1']
        f2 = greeks_file['f2']
        f3 = greeks_file['f3']
        # """

    # if the file is not found, compute them here.
    except:

        if model_kw == 'phi4':

            [Q, f1, f2, f3] = get_greeks_second_order(length, N, dt, A, model_kw)

        else:

            if not absorbing_layer:

                Q1 = get_Q1(N, dt, A, model_kw)
                R2 = get_R2(N, dt, A, model_kw)

                [Q, f1, f2, f3] = get_greeks_first_order(N, dt, A, model_kw)

            elif absorbing_layer:

                Q1 = get_Q1(N, 0.5 * dt, A, model_kw)
                R2 = get_R2(N, 0.5 * dt, A, model_kw)
                [Q, f1, f2, f3] = get_greeks_first_order(N, 0.5 * dt, A, model_kw)

    greeks = dict([('Q', Q), ('f1', f1), ('f2', f2), ('f3', f3)])

    if model_kw == 'phi4':

        propagator = linalg.expm(A.multiply(dt))
        propagator2 = linalg.expm(A.multiply(0.5 * dt))

    else:

        if not absorbing_layer:

            propagator = np.exp(A * dt)
            propagator2 = np.exp(A * 0.5 * dt)

        elif absorbing_layer:

            propagator = np.exp(A * 0.5 * dt)
            propagator2 = np.exp(A * 0.25 * dt)

            damping_mat = assemble_damping_mat(N, length, x, dt)

    Uinit = initial_state

    if model_kw == 'phi4':

        v1 = fft(Uinit[0, :])
        v2 = fft(Uinit[1, :])

        V = np.concatenate((v1, v2))

        # make data storage array
        Udata = np.zeros([2, 1 + int(nsteps / ndump), N], dtype=float)
        Udata[:, 0, :] = Uinit

    else:

        V = fft(Uinit[0, :])

        # make data storage array
        Udata = np.zeros([1 + int(nsteps / ndump), N], dtype=float)
        Udata[0, :] = Uinit[0]

    cnt = 0.  # counter

    for n in np.arange(1, nsteps + 1):

        # TODO: have a user-specified time-stepping order of accuracy in the simulation object
        # Va = do_etdrk1_step(V, propagator, forcing, Q1)
        # Va = do_etdrk2_step(V, propagator, forcing, Q1, R2)
        Va = do_etdrk4_step(V, propagator, propagator2, forcing, greeks)

        if not absorbing_layer:

            #Va = do_ifrk4_step(V, propagator, propagator2, forcing, dt)

            V = Va

        elif absorbing_layer:

            Va = do_ifrk4_step(V, propagator, propagator2, forcing, 0.5*dt)

            Vb = do_diffusion_step(Va, dt, N, damping_mat)

            # V = do_etdrk1_step(Vb, propagator, forcing, Q1)
            # V = do_etdrk2_step(Vb, propagator, forcing, Q1, R2)
            V = do_etdrk4_step(Vb, propagator, propagator2, forcing, greeks)
            #V = do_ifrk4_step(Vb, propagator, propagator2, forcing, 0.5*dt)

            if cnt % 500 == 0:
                pass

                U = np.real(ifft(V))

                U *= 1. - 1. * damping_coeff(-x, length)

                V = fft(U)

        # V = np.reshape(V, (2 * N,))

        cnt += 1

        # print(cnt)

        if cnt % ndump == 0:

            if model_kw == 'phi4':

                Udata[0, int(n / ndump), :] = np.real(ifft(V[0:N]))
                Udata[1, int(n / ndump), :] = np.real(ifft(V[N:]))

            else:

                Udata[int(n / ndump), :] = np.real(ifft(V))

        else:

            pass

    return Udata
