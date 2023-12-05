import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


class Solver():
    def __init__(self, N = 50, L = 1):
        self.N = N
        self.L = L
        self.dx = L/N

        self.x = jnp.linspace(0, L, num = N)
        self.y = jnp.linspace(0, L, num = N)

        yy, xx = jnp.meshgrid(jnp.linspace(0, L, num = N), jnp.linspace(0, L, num = N))
        self.yy = yy
        self.xx = xx

        self.Dx = None
        self.Dy = None
        self.init_differentials()

        self.A = None
        self.B = None
        self.init_boundify()

    def init_differentials(self):
        D = np.zeros(shape = (self.N, self.N))
        for i in range(1, self.N - 1):
            D[i, i - 1] = -1 
            D[i, i + 1] = +1 

            D[0, 0] = -3
            D[0, 1] = 4
            D[0, 2] = -1

            D[-1, -3] = 1
            D[-1, -2] = -4
            D[-1, -1] = 3

        D /= 2*self.dx

        self.Dy = jnp.array(np.kron(np.identity(self.N), D))
        self.Dx = jnp.array(np.kron(D, np.identity(self.N)))

    def init_boundify(self):
        A = np.ones((self.N**2,self.N**2))
        B = np.zeros((self.N**2,self.N**2))

        for i in range(self.N*self.N):
            cond = False

            ## y boundaries
            
            if i % self.N == 0:
                cond = True

            if i % self.N == self.N - 1:
                cond = True


            ## x boundaries

            if i < self.N:
                cond = True

            if i >= self.N*self.N - self.N:
                cond = True

            if cond:
                A[i,:] = np.zeros((self.N*self.N))
                B[i,i] = 1

        self.A = jnp.array(A)
        self.B = jnp.array(B)

    def boundify(self, operator):
        return operator * self.A + self.B

    def solve(self, k, dirichlet):
        k_mat = jnp.diag(k.flatten())
        op = self.Dx@ k_mat @self.Dx + self.Dy@ k_mat@self.Dy
        op = self.boundify(op)

        res = (jnp.linalg.inv(op) @ dirichlet.flatten()).reshape((self.N,self.N))
        return res