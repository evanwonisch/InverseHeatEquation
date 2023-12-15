# InverseHeatEquation

In this project we devise an algorithm to infer the spatially varying conductivity of the heat equation, given just boundary measurements. One starts with defining a conductivity $k(x,y)$ which later has to be inferred and takes multiple measurements of it. This means creating a list of (dirichlet, neumann) boundary value pairs which are satisfied by the conductivity k.

### Chebyshev Solver
The chebyshev solver allows the evaluation of a function $f$ which calculates the neumann boundaries (heat current) of the temperature field given
a conductivity $k$ and dirichlet boundaries:
$$
\mathrm{neumann} = \mathrm{solve}(k, \mathrm{dirichlet})
$$

### Optimize the conductivity
Given the measurements $(\mathrm{dirichlet}_i,\mathrm{neumann}_i)$ one can calulate a loss for a proposed conductivity:
$$
L(k) = \sum_{i} || neumann_i - solve(k, dirichlet_i) ||^2
$$
We optimise this loss by gradient descent.

One can include multiple measurements and thus make the guess more accurate. Se the cleaned up and commented notebooks in "/run", as well as the module.