from absl.testing import absltest, parameterized
from jax_alqp.sparse import Matrix

import jax.numpy as np

import jax

import itertools

from primal_dual_ilqr.linalg_helpers import ldlt


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSparseMatrix(self):
        M_nz = list(itertools.product(range(3), range(3)))
        M_data = np.arange(9, dtype=np.float64)
        M = Matrix(shape=[3, 3], nz=M_nz, data=M_data)
        MTM = M.XTX()
        L, D_diag = MTM.LDLT()
        MTM_sol = np.arange(9).reshape([3, 3]).T @ np.arange(9).reshape([3, 3])
        L_sol, D_diag_sol = ldlt(MTM_sol)
        for i in range(3):
            for j in range(3):
                self.assertTrue(abs(MTM.at((i, j)) - MTM_sol[i, j]) < 1e-6)
                self.assertTrue(abs(L.at((i, j)) - L_sol[i, j]) < 1e-6)
            self.assertTrue(abs(D_diag[i] - D_diag_sol[i]) < 1e-6)
        MTM.data = MTM.data.at[MTM.nz_inv[(0, 0)]].set(MTM.at((0, 0)) + 1.0)
        MTM.data = MTM.data.at[MTM.nz_inv[(1, 1)]].set(MTM.at((1, 1)) + 1.0)
        MTM.data = MTM.data.at[MTM.nz_inv[(2, 2)]].set(MTM.at((2, 2)) + 1.0)
        x = MTM.LDLT_solve(np.array([343.0, 422.0, 501.0]))
        x_sol = np.array([1.0, 2.0, 3.0])
        for i in range(3):
            self.assertTrue(abs(x[i] - x_sol[i]) < 1e-6)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
