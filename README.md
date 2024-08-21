# joe
A library for solving partial differential equations with Fourier spectral discretization in space and high-order exponential time-stepping. 

![hovplot_length=100 5_T=150 0_N=128 0_dt=0 015625_modelkw=ks_ICkw=gaussian_even_alt_nonlinear=True_abslayer=False](https://github.com/user-attachments/assets/68f57405-3af7-42d9-be33-c585b3ba045a)

FEATURES

-Fast, accurate numerical solutions of first and second order in time partial differential equations (PDEs) for real- or complex-valued scalar fields defined on an interval

-Supports periodic boundary conditions as well as absorbing boundaries/sponge layers (to simulate waves going off to spatial infinity)

-Users can either call the PDE they want to simulate from a catalogue of built-in options, or define their own custom PDE.

-Clean, object-oriented approach to handling simulations makes producing visualizations (pictures AND movies) and analysis (ie. accuracy assesment) quite easy 

DEPENDENCIES

numpy, scipy, matplotlib, cmocean (https://matplotlib.org/cmocean/), FFMPEG support for movies

GETTING STARTED

1) Download the zip of the library from github and extract it (of course, you should be a good Python-er and make a new venv with all the dependencies as well, but the base Conda environment has all you need). 
2) Download cmocean for colormaps (conda install conda-forge::cmocean)
3) For processing movies of your numerical solutions in joe, get ffmpeg support. I recommend using the av package (https://pypi.org/project/av/) which you can install with conda install-c conda-forge av.
4) Then, open up the Jupyter tutorial (https://github.com/ageorgemorgan/joe/blob/main/tutorial_START_HERE!.ipynb) to see joe in action. 

FUTURE DIRECTIONS

-Add functionality for fields defined on 2D domains 

-Allow for higher-order-derivatives in time (ie. Timoshenko beam equations)

![hovplot_mod_length=100 0_T=100 0_N=256 0_dt=0 010000_modelkw=focusing_nls_ICkw=nls_soliton_nonlinear=True_abslayer=False](https://github.com/user-attachments/assets/63d2b949-2198-4c5f-ae37-a318df5d3bf7)

