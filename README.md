# joe
A library for solving partial differential equations with Fourier spectral discretization in space and high-order exponential time-stepping. 

![joe](https://github.com/ageorgemorgan/joe/assets/78241397/617e9f7d-af9d-4e49-b046-f442faaa88ab)

(^ Joseph Fourier, the namesake of this project) 

FEATURES

-Fast, accurate numerical solutions of first and second order in time partial differential equations (PDEs) for real scalar fields defined on an interval

-Supports periodic boundary conditions as well as absorbing boundaries/sponge layers (to simulate waves going off to spatial infinity)

-Users can either call the PDE they want to simulate from a catalogue of built-in options, or define their own custom PDE.

-Clean, object-oriented approach to handling simulations makes producing visualizations (pictures AND movies) and analysis (ie. accuracy assesment) quite easy 

DEPENDENCIES

numpy, scipy, matplotlib, cmocean (https://matplotlib.org/cmocean/) 

GETTING STARTED

See the Jupyter notebook here TODO: link to it!!!

FUTURE DIRECTIONS

-Add functionality for complex-valued scalar fields (ie. nonlinear Schroedinger equations)

-Add functionality for fields defined on 2D domains 

-Allow for higher-order-derivatives in time (ie. Timoshenko beam equations)
