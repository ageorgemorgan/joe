import numpy as np
from joe_lab.sponge_layer import clip_spongeless

def test_clip_spongeless():
    N = 32
    z = np.random.rand(N)
    sfrac = 0.5
    assert len(clip_spongeless(z,sfrac)==int(0.5*N))