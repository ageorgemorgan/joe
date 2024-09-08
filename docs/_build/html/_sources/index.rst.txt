.. joe documentation master file, created by
   sphinx-quickstart on Sat Sep  7 15:43:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

joe documentation
===============================

*joe* rapidly generates accurate solutions to partial differential equations for smooth real or complex scalar functions :math:`u(x,t)` that depend on one
spatial variable and one temporal variable.

**FEATURES**

-Fast, accurate numerical solutions of partial differential equations (PDEs) of the form

.. math::
    \partial_{t}^m u +L\left(\frac{1}{i}\partial_{x}\right)u + f(x,t,u,u_{x}, u_{xx},...) = 0,

where :math:`m=1` or :math:`m=2`, :math:`u(x,t)` is a real- or complex-valued scalar field defined on an interval,
and :math:`L(k)` , :math:`f` are some nice functions supplied by the user.

-Supports periodic boundary conditions as well as absorbing boundaries/sponge layers (to simulate waves going off to spatial infinity)

-Users can either call the PDE they want to simulate from a catalogue of built-in options, or define their own custom PDE.

-Easily customizable initial conditions.

-Clean, object-oriented approach to handling simulations makes post-processing (accuracy assesment) very straightforward.

-Producing publication-quality visuals is quick and easy with *joe*'s built-in functions: simply call the right plotting
function on your simulation, tweak a few options, and you've got a plot or movie ready to go. Almost all the required
matplotlib stuff is under the hood.

**DEPENDENCIES**

numpy, scipy, matplotlib, jupyter (for accessing tutorials), cmocean (https://matplotlib.org/cmocean/),
alive-progress (https://pypi.org/project/alive-progress/1.0/). You may also want to download FFmpeg support for
creating movies: see the *joe* GitHub page for tips on how to do this.

**GETTING STARTED**

Open up the Jupyter tutorials (https://github.com/ageorgemorgan/joe/blob/main/demos/tutorials/) to see joe in action!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules.rst

