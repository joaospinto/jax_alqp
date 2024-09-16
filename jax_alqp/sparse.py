from jax import numpy as np

from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Matrix:
    shape: tuple[int, int]
    nz: list[tuple[int, int]]
    nz_inv: dict[tuple[int, int], int]
    data: np.ndarray

    @staticmethod
    def _build_nz_inv(nz: list[tuple[int, int]]):
        out = {}
        for k, ij in enumerate(nz):
            out[ij] = k
        return out

    def __init__(
        self,
        shape: tuple[int, int],
        nz: list[tuple[int, int]],
        data: np.ndarray,
    ):
        assert len(shape) == 2
        assert len(nz) == len(data)
        self.shape = shape
        self.nz = nz
        self.nz_inv = self._build_nz_inv(nz)
        self.data = data

    def at(self, ij: tuple[int, int], default=None):
        data_index = self.nz_inv.get(ij, None)
        if data_index is None:
            return default
        return self.data[data_index]

    def __add__(self, other):
        assert self.shape == other.shape
        nz_set = set(self.nz).union(set(other.nz))
        nz = list(nz_set)
        data = np.zeros(len(nz))
        for k, ij in enumerate(nz):
            data = data.at[k].set(self.at(ij, 0.0) + other.at(ij, 0.0))
        return Matrix(shape=self.shape, nz=nz, data=data)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            assert self.shape[1] == other.shape[0]
            prod_shape = (self.shape[0], other.shape[1])
            # Create the sparsity pattern.
            prod_nz: list[tuple[int, int]] = []
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        if (i, k) in self.nz_inv and (k, j) in other.nz_inv:
                            prod_nz.append((i, j))
            # Fill the data.
            prod_data = np.zeros(len(prod_nz))
            for data_idx, ij in enumerate(prod_nz):
                i, j = ij
                for k in range(self.shape[1]):
                    m = self.nz_inv.get((i, k), None)
                    n = other.nz_inv.get((k, j), None)
                    if m is not None and n is not None:
                        prod_data = prod_data.at[data_idx].set(
                            prod_data[data_idx] + self.data[m] * other.data[n]
                        )

            return Matrix(shape=prod_shape, nz=prod_nz, data=prod_data)

        assert self.shape[1] == other.shape[0]
        assert len(other.shape) == 1
        out = np.zeros([self.shape[0]])
        for k, ij in enumerate(self.nz):
            i, j = ij
            out = out.at[i].set(out[i] + self.data[k] * other[j])
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            data = np.copy(self.data)
            data = other * data
            return Matrix(shape=self.shape, nz=self.nz, data=data)

        assert len(other.shape) == 1
        assert other.shape[0] == self.shape[0]
        data = np.empty(len(self.nz))
        for k, ij in enumerate(self.nz):
            i, _ = ij
            data = data.at[k].set(other[i] * self.data[k])
        return Matrix(shape=self.shape, nz=self.nz, data=data)

    def transpose(self):
        assert len(self.shape) == 2
        shape = [self.shape[1], self.shape[0]]
        nz = [(ij[1], ij[0]) for ij in self.nz]
        return Matrix(shape=shape, nz=nz, data=self.data)

    @property
    def T(self):
        return self.transpose()

    def XTX(self):
        return self.transpose() @ self

    def LDLT(self):
        assert self.shape[0] == self.shape[1]
        n = self.shape[0]
        assert n > 0

        # Note that The L D L^T decomposition can be computed with the following recursion:
        # D_i    = self_ii - sum_{j=0}^{i-1} L_{ij}^2 D_j
        # L_{ij} = (1 / D_j) * (self_{ij} - sum_{k=0}^{j-1} L_{ik} L_{jk} D_k)

        # Compute L_nz
        L_nz_set = set(self.nz)
        for i in range(n):
            L_nz_set.add((i, i))
        for i in range(n):
            for j in range(i):
                for k in range(j):
                    if (i, k) in L_nz_set and (j, k) in L_nz_set:
                        L_nz_set.add((i, j))
        L_nz = list(L_nz_set)

        # Create L and D_diag.
        L = Matrix(shape=(n, n), nz=L_nz, data=np.zeros(len(L_nz)))
        D_diag = np.zeros(n)

        # Fill L and D_diag via the recursion rule above.
        for i in range(n):
            L.data = L.data.at[L.nz_inv[(i, i)]].set(1.0)
            D_diag = D_diag.at[i].set(self.at((i, i), 0.0))

        for i in range(n):
            for j in range(i):
                if (i, j) not in L_nz_set:
                    continue
                data_idx = L.nz_inv[(i, j)]
                x = self.at((i, j))
                if x is not None:
                    L.data = L.data.at[data_idx].set(x)
                for k in range(j):
                    if (i, k) in L_nz_set and (j, k) in L_nz_set:
                        L.data = L.data.at[data_idx].set(
                            L.data[data_idx]
                            - D_diag[k]
                            * L.data[L.nz_inv[(i, k)]]
                            * L.data[L.nz_inv[(j, k)]]
                        )
                L.data = L.data.at[data_idx].set(L.data[data_idx] / D_diag[j])
            for j in range(i):
                if (i, j) not in L_nz_set:
                    continue
                D_diag = D_diag.at[i].set(
                    D_diag[i] - L.data[L.nz_inv[(i, j)]] ** 2 * D_diag[j]
                )

        return L, D_diag

    def zero_out_row(self, row):
        for k, ij in enumerate(self.nz):
            i, j = ij
            if i == row:
                self.data = self.data.at[k].set(0.0)

    def LDLT_solve(self, rhs):
        L, D_diag = self.LDLT()
        L_T = L.transpose()
        n = self.shape[0]
        # Solve L z = rhs.
        z = np.empty(n)
        for i in range(n):
            z = z.at[i].set(rhs[i])
            for j in range(i):
                z = z.at[i].set(z[i] - L.at((i, j), 0.0) * z[j])
        # Solve D y = z
        y = np.empty(n)
        for i in range(n):
            y = y.at[i].set(z[i] / D_diag[i])
        # Solve L^T x = y
        x = np.empty(n)
        for i in range(n - 1, -1, -1):
            x = x.at[i].set(y[i])
            for j in range(n - 1, i, -1):
                x = x.at[i].set(x[i] - L_T.at((i, j), 0.0) * x[j])
        return x

    def todense(self):
        out = np.zeros(self.shape)
        for k, ij in enumerate(self.nz):
            i, j = ij
            out = out.at[i, j].set(self.data[k])
        return out

    @staticmethod
    def fromdense(M):
        assert len(M.shape) == 2
        nz = []
        data = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] != 0.0:
                    nz.append((i, j))
                    data.append(M[i, j])
        return Matrix(shape=M.shape, nz=nz, data=np.array(data))

    def tree_flatten(self):
        children = self.data
        aux_data = (self.shape, self.nz, self.nz_inv)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        shape, nz, nz_inv = aux_data
        data = children
        return Matrix(shape=shape, nz=nz, data=data)
