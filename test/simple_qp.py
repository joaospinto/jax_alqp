from absl.testing import absltest, parameterized
from jax_alqp.solver import solve_in_osqp_format, sparse_solver_exporter

import numpy as np

import jax


class Test(parameterized.TestCase):
    def setUp(self):
        super(Test, self).setUp()

    def testSimpleQP(self):
        with jax.disable_jit():
            # Simple test from the OSQP repo.
            P = np.array([[4.0, 1.0], [1.0, 2.0]])
            q = np.ones(2)

            A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
            lower = np.array([1.0, 0.0, 0.0, -np.inf])
            upper = np.array([1.0, 0.7, 0.7, np.inf])

            n = P.shape[0]

            ws_x = np.zeros(n)

            x, iteration_al, succeeded = solve_in_osqp_format(
                P=P, q=q, A=A, l=lower, u=upper, ws_x=ws_x
            )

            if succeeded:
                print(f"Solved successfully in {iteration_al} iterations.")
            else:
                print(f"Failed to solve in {iteration_al} iterations.")

            self.assertTrue(succeeded)

            self.assertTrue(np.linalg.norm(x - np.array([0.3, 0.7]) < 1e-6))

    def testSparseQP(self):
        with jax.disable_jit():
            exported_solver = sparse_solver_exporter(
                P=np.ones(4).reshape([2, 2]),
                C=np.ones(2).reshape([1, 2]),
                G=np.ones(8).reshape([4, 2]),
            )

            x, y_eq, y_ineq, eq, ineq, iteration_al, succeeded = (
                exported_solver(
                    P_data=np.array([4.0, 1.0, 1.0, 2.0]),
                    q=np.ones(2),
                    C_data=np.array([1.0, 1.0]),
                    d=np.array([1.0]),
                    G_data=np.array([-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0]),
                    h=np.array([0.0, 0.0, 0.7, 0.7]),
                    ws_x=np.zeros(2),
                )
            )

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
    absltest.main()
