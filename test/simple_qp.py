from absl.testing import absltest, parameterized
from jax_alqp.solver import solve_in_osqp_format

import numpy as np
from scipy import sparse as sp

import jax


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSimpleQP(self):
        # Simple test from the OSQP repo.
        P = sp.triu([[4.0, 1.0], [1.0, 2.0]], format="csc")
        q = np.ones(2)

        A = sp.csc_matrix(
            np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        )
        l = np.array([1.0, 0.0, 0.0, -np.inf])
        u = np.array([1.0, 0.7, 0.7, np.inf])

        n = P.shape[0]
        m = A.shape[0]

        ws_x = np.zeros(n)

        x, iteration_al, succeeded = solve_in_osqp_format(P, q, A, l, u, ws_x)

        if succeeded:
            print(f"Solved successfully in {iteration_al} iterations.")
        else:
            print(f"Failed to solve in {iteration_al} iterations.")

        self.assertTrue(succeeded)

        self.assertTrue(np.linalg.norm(x - np.array([0.3, 0.7]) < 1e-6))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    np.set_printoptions(threshold=1000000)
    np.set_printoptions(linewidth=1000000)

    absltest.main()
