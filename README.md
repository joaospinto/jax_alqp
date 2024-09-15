This repository implements a simple QP solver using an Augmented Lagrangian method.
Specifically, problems of the following form are handled (with JIT support):

$$\min\limits_{x} 0.5 x^T P x + q^T x \qquad \mbox{s.t.} \quad Cx = d \wedge Gx <= h$$

We also support problems of the following form:

$$\min\limits_{x} 0.5 x^T P x + q^T x \qquad \mbox{s.t.} \quad l <= Ax <= u$$

In both cases, we require that $P$ be symmetric positive semi-definite,
and that $P + \rho C^T C$ be positive definite for any $\rho > 0$.
We should eventually support only requiring $P$ to be symmetric positive semi-definite,
by adding a small amount of regularization (similarly to what OSQP does).

My motivation for writing this simple solver is that OSQP often struggles to solve problems
to high accuracy in a reasonable number of iterations. It would be interesting to benchmark
this simple solver against PIQP, and possibly implementing the latter in JAX.
