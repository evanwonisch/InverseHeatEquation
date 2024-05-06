The problem we will try to tackle in this project is the following: imagine we are given some object and are tasked to determine its internal heat conductivity without destroying it. What we can do experimentally is to heat up the system and measure surface temperature as well as flux through the surface, and using this measurements we can learn about heat conductivity inside of the object. In mathematics this problem is formally known as Caldéron problem \cite{paper}, and in the essence, it is the problem of solving the heat equation with the unknown spatial distribution of conductivity with both Dirichlet and von-Neumann boundary conditions given.\\
First thing that one asks oneself is the uniqueness of the solution to the problem, for given boundary conditions. Interestingly enough, there is a proof of uniqueness which states that the conductivity $k$ can be uniquely determined if all possible pairs of temperature and heat flux are known. In a physical setting, this is unrealistic, as one has only access to a finite number of measurements. Thus, imposing certain regularity conditions on the conductivity and a finite number of measurements could be used to generate sensible estimates. Thus we pose the following questions:
- Can an algorithm be devised to infer the conductivity $k$ from given measurements?

- What assumptions can be made to limit the degeneracy of the problem?

- With what accuracy can $k$ be inferred?

- How many measurements are needed to find a reliable estimate?
    
## Forward Problem
The heat equation describes the spatial dependence of temperature $T$, given a local heat conductivity $k(\vec{r})$. In steady state, it is written as:

$$
    \nabla \cdot (k(\vec{r})\, \nabla T) = 0
$$

In this problem, the area of interest will be taken as subset $\Omega = [-1,\, 1]^2 \subset \mathbb{R}^2$.
Solving this equation allows for specification of  either Dirichlet or von-Neumann boundaries:

$$
    T|_{\delta \Omega} = f \quad\quad \text{Dirichlet} 
    $$

$$
    (k \cdot \nabla T)|_{\delta \Omega}\cdot \vec{n} = g \quad\quad \text{von-Neumann}
$$

$f, g \in C^{\infty}(\delta \Omega)$ are functions on the boundary and $\vec{n}$ is the surface normal vector.
A map $\Lambda_k$ guiding from Dirichlet to von-Neumann boundaries is introduced:

$$
    \Lambda_k : C^{\infty}(\delta \Omega) \to C^{\infty}(\delta \Omega) \\
    \Lambda_k f = g
$$

To evaluate this map, a numerical algorithm is implemented to compute the solution of $T$ for a given conductivity field $k$ and Dirichlet condition $f$. Having obtained a solution, the resulting von-Neumann boundaries can be obtained.

## Inverse Problem
Having obtained a finite number of measurements $(f_i, g_i)$, that being pairs of Dirichlet and von-Neumann boundary conditions, we seek to find the conductivity $k$ which satisfies all of them as best as possible. Therefore, a loss function $L$ is introduced:

$$
    L(k) = \sum_i ||g_i  - \Lambda_k f_i||^2
$$

Here, the sum goes over all obtained measurements and the norm $||\cdot||$ is the regular norm of boundary valued functions:

$$
    ||h||^2 = \int_{\delta \Omega} |h(r)|^2 dr
$$

A simple optimisation scheme is now applied to optimise the above loss function to ideally obtain $k_\ast$ from which the measurements originate:

$$
    k_\ast \in \underset{k \in \mathcal{H}}{\mathrm{argmin}} \sum_i ||g_i  - \Lambda_k f_i||^2
$$

Here, a class of possible conductivities $\mathcal{H}$ will be chosen. It is not a priory clear if the above problem is a convex optimisation problem (i.e. having a unique minimum) and its degeneracy is later studied. 
