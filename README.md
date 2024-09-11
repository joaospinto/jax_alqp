This repository implements a simple QP solver using an Augmented Lagrangian method.
Specifically, problems of the following form are handled (with JIT support):

$$\min\limits_{x} 0.5 x^T P x + q^T x \qquad \mbox{s.t.} \quad Cx = d \wedge Gx <= h$$

We also support problems of the following form:

$$\min\limits_{x} 0.5 x^T P x + q^T x \qquad \mbox{s.t.} \quad l <= Ax <= u$$

In both cases, we require that $P$ be symmetric positive semi-definite.
The current implementation currently uses a Cholesky decomposition,
instead of $L D L^T$, due to JAX's lack of native support of the latter.
I will address this limitation shortly, but for now it means $P$ should be symmetric positive definite.

My motivation for writing this simple solver is that OSQP often struggles to solve problems
to high accuracy in a reasonable number of iterations.
