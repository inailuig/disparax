import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
import numpy as np
from functools import partial, lru_cache


@lru_cache
def prep_data(m, ndevices, col_major=False):
    """
    prepare the internal datastructures for the lower-triangular 2d block-cyclic representation

    the triangular part is concatenated, and then distributed device after device in stripes:
    row major:  col major:
       [0 X X]     [0 X X]
       [1 2 X]     [1 0 X]
       [0 1 2]     [2 1 2]

    Args:
        m: number of blocks in each row / column
        col_major: distribute the blocks in cloumn-major or row-major order
    Returns:
        ij: indices of the rows and cols in the flattened data (without padding)
        npad: how many elements to add to make ij divisible by the number of devices
        data_row:
            dev_row_ind_diag: for every device and row the index of the diagonal element in the device's data if present
            dev_row_ind_offdiag_data: for every device and row the index of the off-diagonal elements in the device's data if present
            dev_row_ind_offdiag_col: for every device and row the index of column of the off-diagonal elements in the device's data if present
            dev_ind_diag : for every row/col the index of the device  which contains the diagonal
        data_col:
            same elements as data_row but for the columns
    """
    if col_major:
        ij = np.array(np.triu_indices(m))[::-1].T
    else:
        ij = np.array(np.tril_indices(m)).T

    npad = int(len(ij) // ndevices + 1) * ndevices - len(ij)

    pad_val = m + 1
    dev_distr = np.full((m, m), dtype=int, fill_value=pad_val) + np.tril(np.ones((m, m), dtype=int))
    for k, (i, j) in enumerate(ij):
        dev_distr[i, j] = k % ndevices

    dev_ind_diag = dev_distr.diagonal()
    dev_distr_offdiag = dev_distr - np.diag(dev_ind_diag - pad_val)

    # max number of off-diagonal entries per row, of any device
    maxperrow = max(np.bincount(r)[:ndevices].max() for r in dev_distr_offdiag)
    maxpercol = max(np.bincount(r)[:ndevices].max() for r in dev_distr_offdiag.T)

    dev_row_ind_offdiag_col = np.full((ndevices, m, maxperrow), dtype=int, fill_value=-1)
    dev_row_ind_offdiag_data = np.full((ndevices, m, maxperrow), dtype=int, fill_value=-1)
    dev_row_ind_diag = np.full((ndevices, m), dtype=int, fill_value=-1)

    dev_col_ind_offdiag_row = np.full((ndevices, m, maxpercol), dtype=int, fill_value=-1)
    dev_col_ind_offdiag_data = np.full((ndevices, m, maxpercol), dtype=int, fill_value=-1)
    dev_col_ind_diag = np.full((ndevices, m), dtype=int, fill_value=-1)

    for d in range(ndevices):
        for row in range(m):
            (cols,) = np.where(dev_distr_offdiag[row] == d)
            dev_row_ind_offdiag_col[d, row, : len(cols)] = cols

    for d in range(ndevices):
        for col in range(m):
            (rows,) = np.where(dev_distr_offdiag[:, col] == d)
            dev_col_ind_offdiag_row[d, col, : len(rows)] = rows

    all_ind = np.zeros((m, m), dtype=int)
    if col_major:
        for d in range(ndevices):
            k = 0
            for col in range(m):
                if dev_distr[col, col] == d:
                    all_ind[col, col] = k
                    k = k + 1
                for row in range(col + 1, m):
                    if dev_distr[row, col] == d:
                        all_ind[row, col] = k
                        k = k + 1
    else:  # row major
        for d in range(ndevices):
            k = 0
            for row in range(m):
                for col in range(row):
                    if dev_distr[row, col] == d:
                        all_ind[row, col] = k
                        k = k + 1
                if dev_distr[row, row] == d:
                    all_ind[row, row] = k
                    k = k + 1

    for d in range(ndevices):
        for row in range(m):
            row_tmp = []
            for col in range(row):
                if dev_distr[row, col] == d:
                    row_tmp.append(all_ind[row, col])
                dev_row_ind_offdiag_data[d, row, : len(row_tmp)] = row_tmp
            if dev_distr[row, row] == d:
                dev_row_ind_diag[d, row] = all_ind[row, row]

    for d in range(ndevices):
        for col in range(m):
            col_tmp = []
            if dev_distr[col, col] == d:
                dev_col_ind_diag[d, col] = all_ind[col, col]
            for row in range(col + 1, m):
                if dev_distr[row, col] == d:
                    col_tmp.append(all_ind[row, col])
                dev_col_ind_offdiag_data[d, col, : len(col_tmp)] = col_tmp

    data_row = (dev_row_ind_diag, dev_row_ind_offdiag_data, dev_row_ind_offdiag_col, dev_ind_diag)
    data_col = (dev_col_ind_diag, dev_col_ind_offdiag_data, dev_col_ind_offdiag_row, dev_ind_diag)

    return ij, npad, data_row, data_col


def pack_blocks(M, bs, ndevices, col_major=False, axis_name="i"):
    """
    pack a replicated hermitian or lower triangular matrix, pad to number of devices, numpy version intended for testing only
    -> please use compute_blocks to construct the packed matrix directly

    Args:
        M: hermitian/symmetric matrix A or lower triangular matrix L; only the blocks on the lower triangles are accessed
        bs: block size
        col_major: whether M is in column major or row-major representation
    Returns:
        lower-triangular 2d block-cyclic representation of M
    """
    m = M.shape[0] // bs
    Mb = M.reshape(m, bs, m, bs).transpose(0, 2, 1, 3)

    ij, npad, _, _ = prep_data(m, ndevices, col_major=col_major)

    block_data = np.array([Mb[i, j] for i, j in ij])
    block_data = np.pad(block_data, [(0, npad), (0, 0), (0, 0)])
    block_data = block_data.reshape(-1, ndevices, *block_data.shape[1:]).transpose(1, 0, 2, 3)
    block_data = jax.lax.collapse(block_data, 0, 2)
    return reshard(block_data, P(axis_name))


@partial(jax.jit, static_argnames=("bs", "col_major", "axis_name", "batch_size"))
def compute_blocks(f, x, bs, col_major=False, axis_name="i", batch_size=0):
    """
    compute a symmetric/hermitian matrix A in blocks (lower triangular only)
    Args:
        f: a vectorized function f(x,y) computing A_ij = f[xi,xj]
          - if hermitian the funciton needs to conjugate x internally
          - the function needs wrapped in a jax.tree_util.Partial
        x: inputs; can be a pytree of arrays with equal lenght first axis
        bs: block size, needs to divide first axis of x
    Returns:
        lower-triangular 2d block-cyclic representation of A

    Example:
        construct the block representation from the replicated dense matrix
        ```python
        import jax
        import jax.numpy as jnp
        from jax.tree_util import Partial
        from jax.sharding import Mesh, AxisType, set_mesh
        import disparax as dpx

        set_mesh(Mesh(jax.devices(), "i", axis_types=(AxisType.Explicit)))

        n = 32
        bs = 8
        A_dense = jnp.eye(n)

        def pack_blocks(A_dense, bs, col_major=False, axis_name="i")
            x = jnp.arange(n)
            f = Partial(lambda A, i, j: A[i, :][:, j], A_dense)
            return dpx.compute_blocks(f, x, bs=bs, col_major=True, axis_name=axis_name)

        A = pack_blocks(A_dense, bs, col_major=False, axis_name="i")
        ```
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=0, in_axes=(None, None))
    def _compute_blocks(f, x):
        n = len(jax.tree.leaves(x)[0])
        m = n // bs
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        x_blocks = jax.tree.map(lambda x: x.reshape(m, bs, *x.shape[1:]), x)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
        pad_mask_padded[-npad:] = False
        # ij: index of the blocks on the current device; mask: wether they are padding or not
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]
        pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]

        def _f(args):
            i, j, m = args
            xi = jax.tree.map(lambda x: x[i], x_blocks)
            xj = jax.tree.map(lambda x: x[j], x_blocks)
            res = f(xi, xj)
            return jax.tree.map(lambda r: jnp.where(m, r, jnp.zeros_like(r)), res)

        return jax.lax.map(_f, (*ij.T, pad_mask), batch_size=batch_size)

    return _compute_blocks(f, x)
