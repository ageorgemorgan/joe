# joe
A library for solving partial differential equations for real or complex scalar functions $u(x,t)$ that depend on one spatial variable and one temporal variable. *joe* uses Fourier spectral discretization in space and high-order exponential time-stepping to rapidly and precisely solve initial-value problems. 

FEATURES

-Fast, accurate numerical solutions of partial differential equations (PDEs) of the form 
$$
\partial_{t}^m u +L\left(\frac{1}{i}\partial_{x}\right)u + f(x,t,u,u_{x}, u_{xx},...) = 0,
$$
where $m=1$ or $m=2$, $u(x,t)$ is a real- or complex-valued scalar field defined on an interval, and $L(k)$ , $f$ are some nice functions supplied by the user.

-Supports periodic boundary conditions as well as absorbing boundaries/sponge layers (to simulate waves going off to spatial infinity)

-Users can either call the PDE they want to simulate from a catalogue of built-in options, or define their own custom PDE.

-Easily customizable initial conditions.

-Clean, object-oriented approach to handling simulations makes post-processing (accuracy assesment) very straightforward.      

-Producing publication-quality visuals is quick and easy with *joe*'s built-in functions: simply call the right plotting function on your simulation, tweak a few options, and you've got a plot or movie ready to go. Almost all the required matplotlib stuff is under the hood.  

DEPENDENCIES

numpy, scipy, matplotlib, cmocean (https://matplotlib.org/cmocean/), alive-progress (https://pypi.org/project/alive-progress/1.0/). You may also want to download FFmpeg support for creating movies: I recommend using the PyAV package (https://pypi.org/project/av).

GETTING STARTED
 
Open up the Jupyter tutorial (https://github.com/ageorgemorgan/joe/blob/main/demos/tutorial_START_HERE!.ipynb) to see joe in action! More tutorials are coming soon. 

Currently getting FFmpeg support is a bit of a tough one and it *does not auto-install when you install joe*. I recommend using conda, installing *joe* in a conda environment, and then using conda to install PyAV via 

```
conda install av
```

or 

```
conda install av -c conda-forge
```

FUTURE DIRECTIONS

-Get support for movies on install without installing further packages!

-Add functionality for fields defined on 2D domains 

-Allow for higher-order-derivatives in time (ie. Timoshenko beam equations)
