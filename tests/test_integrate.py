import numpy as np
import joe_lab.utils as joe_utils

def test_integrate():
    length = 2.*np.pi
    N = 32
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    u = np.sin(3.*x)
    integral = joe_utils.integrate(u,length)
    assert np.abs(integral)<1e-15