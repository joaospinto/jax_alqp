import jax.numpy as np
import jax
from functools import partial

import numpy as onp
import timeit


@jax.jit
def solve(
    P,
    q,
    C,
    d,
    G,
    h,
    ws_x,
    *,
    max_al_iterations=10,
    c_threshold=1e-6,
    complementary_slackness_threshold=1e-6,
    penalty_init=1.0,
    penalty_update_rate=10.0,
):
    """
    Solves a QP of the form:
        min_x (0.5 x^T P x + q^T x) s.t. (Cx = d and Gx <= h)
    This method supports JIT compilation.
    """

    def evaluate_constraints(x):
        eq = C @ x - d
        ineq = G @ x - h
        return eq, ineq

    def augmented_lagrangian(x, y_eq, y_ineq, penalty):
        J = 0.5 * (P @ x).T @ x + q @ x
        eq, ineq = evaluate_constraints(x)
        active_set = np.invert(np.isclose(y_ineq, 0.0) & (ineq < 0.0))
        J += y_eq.T @ eq + 0.5 * penalty * eq.T @ eq
        J += y_ineq.T @ ineq + 0.5 * penalty * ineq.T @ (active_set * ineq)
        return J

    def dual_update(constraint, dual, penalty):
        return dual + penalty * constraint

    def body(inputs):
        (
            x,
            y_eq,
            y_ineq,
            penalty,
            eq,
            ineq,
            iteration_al,
            _unused_succeeded,
            _unused_should_continue,
        ) = inputs

        # Update primal variables
        al = partial(
            augmented_lagrangian, y_eq=y_eq, y_ineq=y_ineq, penalty=penalty
        )

        lhs = jax.hessian(al)(x)
        rhs = -jax.grad(al)(x)

        f = jax.scipy.linalg.cho_factor(lhs)
        dx = jax.scipy.linalg.cho_solve(f, rhs)
        x = x + dx

        # Evaluate constraints
        eq, ineq = evaluate_constraints(x)

        # Update dual variables
        y_eq = dual_update(eq, y_eq, penalty)
        y_ineq = np.maximum(dual_update(ineq, y_ineq, penalty), 0.0)

        def solve_status():
            max_constraint_violation = np.maximum(
                np.max(np.abs(eq)),
                np.max(np.maximum(ineq, 0.0)),
            )

            max_complementary_slack = np.max(np.abs(ineq * y_ineq))

            it_ok = iteration_al < max_al_iterations

            succeeded = np.logical_and(
                max_constraint_violation <= c_threshold,
                max_complementary_slack <= complementary_slackness_threshold,
            )

            should_continue = np.logical_and(
                it_ok,
                np.logical_not(succeeded),
            )

            return succeeded, should_continue

        succeeded, should_continue = solve_status()

        return (
            x,
            y_eq,
            y_ineq,
            penalty * penalty_update_rate,
            eq,
            ineq,
            iteration_al + 1,
            succeeded,
            should_continue,
        )

    def continuation_criteria(inputs):
        (
            _unused_x,
            _unused_y_eq,
            _unused_y_ineq,
            _unused_penalty,
            _unused_eq,
            _unused_ineq,
            _unused_iteration_al,
            _unused_succeeded,
            should_continue,
        ) = inputs

        return should_continue

    x_in = ws_x
    eq_in, ineq_in = evaluate_constraints(x_in)
    y_eq_in = np.zeros_like(eq_in)
    y_ineq_in = np.zeros_like(ineq_in)
    iteration_il_in = 0
    succeeded_in = False
    should_continue_in = True

    (
        x,
        y_eq,
        y_ineq,
        penalty,
        eq,
        ineq,
        iteration_al,
        succeeded,
        _unused_should_continue,
    ) = jax.lax.while_loop(
        continuation_criteria,
        body,
        (
            x_in,
            y_eq_in,
            y_ineq_in,
            penalty_init,
            eq_in,
            ineq_in,
            iteration_il_in,
            succeeded_in,
            should_continue_in,
        ),
    )

    return x, y_eq, y_ineq, eq, ineq, iteration_al, succeeded


def solve_in_osqp_format(
    P,
    q,
    A,
    l,
    u,
    ws_x,
    *,
    max_al_iterations=10,
    c_threshold=1e-6,
    complementary_slackness_threshold=1e-6,
    penalty_init=1.0,
    penalty_update_rate=10.0,
):
    """
    Solves a QP of the form:
        min_x (0.5 x^T P x + q^T x) s.t. (l <= Ax <= u)
    This method does not currently support JIT compilation.
    """
    idx_eq = l == u

    P = P.todense()
    A = A.todense()

    C = A[idx_eq, :]
    d = u[idx_eq]

    idx_ineq_left = onp.logical_and(onp.isfinite(l), onp.logical_not(idx_eq))
    idx_ineq_right = onp.logical_and(onp.isfinite(u), onp.logical_not(idx_eq))

    A_left_ineq = A[idx_ineq_left, :]
    l_left_ineq = l[idx_ineq_left]

    A_right_ineq = A[idx_ineq_right, :]
    u_right_ineq = u[idx_ineq_right]

    G = onp.concatenate([-A_left_ineq, A_right_ineq])
    h = onp.concatenate([-l_left_ineq, u_right_ineq])

    P_in = np.array(P)
    C_in = np.array(C)
    G_in = np.array(G)

    q_in = np.array(q)
    d_in = np.array(d)
    h_in = np.array(h)
    ws_x_in = np.array(ws_x)

    outputs = solve(
        P=P_in,
        q=q_in,
        C=C_in,
        d=d_in,
        G=G_in,
        h=h_in,
        ws_x=ws_x_in,
        max_al_iterations=max_al_iterations,
        c_threshold=c_threshold,
        complementary_slackness_threshold=complementary_slackness_threshold,
        penalty_init=penalty_init,
        penalty_update_rate=penalty_update_rate,
    )

    x, y_eq, y_ineq, eq, ineq, iteration_al, succeeded = outputs

    return x, iteration_al, succeeded
