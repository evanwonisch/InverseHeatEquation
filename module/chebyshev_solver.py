import jax
import jax.numpy as jnp
import numpy as np


class Solver():
    def __init__(self, N = 50):
        self.N = N

        self.grid = np.cos((2*np.arange(N) + 1)*np.pi/2/N)

        self.yy, self.xx = np.meshgrid(self.grid, self.grid)


        self.I = None
        self.Dx = None
        self.Dy = None
        self.D2x = None
        self.D2y = None
        self.DxDy = None
        self.init_differentials()

        self.A = None
        self.B = None
        self.init_boundify()

    def init_differentials(self):
        

        #
        #  Unity Matrix
        #

        I_ = np.zeros((self.N, self.N))

        for i in range(self.N):
            cat = np.zeros(self.N)
            cat[i] = 1
            I_[:, i] = np.polynomial.chebyshev.chebval(self.grid, cat)

        #
        # Differetiation Matrix
        #

        D_ = np.zeros((self.N, self.N))

        for i in range(self.N):
            cat = np.zeros(self.N)
            cat[i] = 1

            Cheb = np.polynomial.chebyshev.Chebyshev(cat)
            D_[:, i] = Cheb.deriv(m = 1)(self.grid)


        #
        # Differetiation Matrix
        #

        D2_ = np.zeros((self.N, self.N))

        for i in range(self.N):
            cat = np.zeros(self.N)
            cat[i] = 1

            Cheb = np.polynomial.chebyshev.Chebyshev(cat)
            D2_[:, i] = Cheb.deriv(m = 2)(self.grid)


        ###
        ###  Use Kronecker Products
        ###

        self.I = jnp.kron(I_, I_)
        self.Dx = jnp.kron(D_, I_)
        self.Dy = jnp.kron(I_, D_)
        self.D2x = jnp.kron(D2_, I_)
        self.D2y = jnp.kron(I_, D2_)
        self.DxDy = jnp.kron(D_, D_)

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
                B[i,:] = self.I[i, :]

        self.A = jnp.array(A)
        self.B = jnp.array(B)

    def boundify(self, operator):
        return operator * self.A + self.B

    def solve(self, k_cheby, dirichlet):
        """
        shape k_cheby = (N**2,) (flattened chebyshev coefficients)
        shape dirichelt = (N, N) (physical space)
        """
        k = self.I @ k_cheby
        kx = self.Dx @ k_cheby
        ky = self.Dy @ k_cheby
        k_mat = np.diag(k)
        kx_mat = np.diag(kx)
        ky_mat = np.diag(ky)

        L = self.boundify(kx_mat @ self.Dx + ky_mat @ self.Dy + k_mat @ (self.D2x + self.D2y))
        sol = np.linalg.inv(L) @ dirichlet.flatten()

        result = {"T":self.I @ sol, "dx T":self.Dx @ sol, "dy T":  self.Dy @ sol, "k": k}
        return result