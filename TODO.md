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