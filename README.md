# joe
A library for solving partial differential equations with Fourier spectral discretization in space and high-order exponential time-stepping. 

FEATURES

-Fast, accurate numerical solutions of first and second order in time partial differential equations (PDEs) for real scalar fields defined on an interval

-Supports periodic boundary conditions as well as absorbing boundaries/sponge layers (to simulate waves going off to spatial infinity)

-Users can either call the PDE they want to simulate from a catalogue of built-in options, or define their own custom PDE.

-Clean, object-oriented approach to handling simulations makes producing visualizations (pictures AND movies) and analysis (ie. accuracy assesment) quite easy 

DEPENDENCIES

numpy, scipy, matplotlib, cmocean (https://matplotlib.org/cmocean/) 

GETTING STARTED

Simply download the zip of the library from github and extract it (of course, you should be a good Python-er and make a new venv with all the dependencies as well). Then, open up the Jupyter tutorial (https://github.com/ageorgemorgan/joe/blob/main/tutorial_START_HERE!.ipynb) to see joe in action. 

FUTURE DIRECTIONS

-Add functionality for complex-valued scalar fields (ie. nonlinear Schroedinger equations)

-Add functionality for fields defined on 2D domains 

-Allow for higher-order-derivatives in time (ie. Timoshenko beam equations)

KNOWN ISSUES

-ETDRK4 doesn't quite work with KdV, but IFRK4 seems to work fine. So, if using the code to solve KdV make sure you
write " method_kw='ifrk4' " when you run a simulation. 

-Users have said their installation of LaTeX (which joe uses to make labels on graphs) doesn't like to cooperate with matplotlib, so in the Aug. 23 version no visuals could be produced. I will put a band-aid fix on this soon!
