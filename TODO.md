## TODO

### 1. numerically check chebyshev accuracy
- accurate

### 2.  match boundary conditions (find a k field)
- optimisation procedure
- gradient point
- positive hessian

### 3. loss landscape (gaussian)
- positive semidefinite
    - possible canyons
    - possible machine precision

- check exponential decay

- check banana stuff

### 4. uniqueness (include multi measurements)
- plot: number of measurements and information gain


### 5. if time permits
- information theory
- gridsize/number of measurements


# Table of Contents

## 1. Introduction
- physical and math problem

## 2. Chebyshev solver
- (k + dirichlet) -> neumann
- pseudospectral method
- implementation
- accuracy comparison

## 3.Inverse Problem
- show the entire process:
    - generate k
    - generate measurements
    - parametrise inferred k
    - define loss function as |\Delta neumann|^2
    - optimise with gradient descent (alpha = 0.003, N = 2000)
    - retrieve inferred k and compare

# 4. Results
- plots:
    - show neumanns, dirichlet
    - nice 20x20 visuals
    - error depending in #measurements
    - runner convergence / Hessian pos. def.

# 5. Discussion
- outlook/problems summary

### Appendices:
- bananas
- varied index decay