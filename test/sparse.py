from absl.testing import absltest, parameterized
from jax_alqp.sparse import Matrix

import jax.numpy as np

import jax

import itertools


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSparseAddition(self):
        A = Matrix.fromdense(np.array([[1.0, 0.0], [0.0, 4.0]]))
        B = Matrix.fromdense(np.array([[0.0, 2.0], [3.0, 0.0]]))
        C = A + B
        C_sol = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertTrue(np.linalg.norm(C.todense() - C_sol) < 1e-6)

    def testSparseElementMultiplication(self):
        A_dense = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([1.0, 0.0])
        A = Matrix.fromdense(A_dense)
        C = v * A
        C_sol = v.reshape([2, 1]) * A_dense
        self.assertTrue(np.linalg.norm(C.todense() - C_sol) < 1e-6)

    def testSparseMatrixMultiplication(self):
        A_dense = np.array([[1.0], [1.0]])
        B_dense = np.array([[1.0, 1.0]])
        A = Matrix.fromdense(A_dense)
        B = Matrix.fromdense(B_dense)
        C = A @ B
        C_sol = A_dense @ B_dense
        self.assertTrue(np.linalg.norm(C.todense() - C_sol) < 1e-6)

    def testSparseXTX(self):
        M_nz = list(itertools.product(range(3), range(3)))
        M_data = np.arange(9, dtype=np.float64)
        M = Matrix(shape=(3, 3), nz=M_nz, data=M_data)
        MTM = M.XTX()
        MTM_sol = np.arange(9).reshape([3, 3]).T @ np.arange(9).reshape([3, 3])
        self.assertTrue(np.linalg.norm(MTM.todense() - MTM_sol) < 1e-6)

    def testSparseLDLTSolve(self):
        MTM = Matrix.fromdense(
            np.arange(9).reshape([3, 3]).T @ np.arange(9).reshape([3, 3])
            + np.eye(3)
        )
        x = MTM.LDLT_solve(np.array([343.0, 422.0, 501.0]))
        x_sol = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.linalg.norm(x - x_sol) < 1e-6)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
