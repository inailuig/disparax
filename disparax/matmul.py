import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
import numpy as np
from functools import partial, lru_cache

from .data import prep_data


@partial(jax.jit, static_argnames=("col_major", "axis_name", "mode"))
def _block_matmul(M, b, col_major=False, axis_name="i", mode="lower"):
    """Matrix-vector and Matrix-matrix multiplication A @ b, L@b and L^H @ b
    Args:
         M: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A or lower triangular matrix L, of mxm blocks of size bs
         b: dense vector/matrix where the first axis is in blocks, i.e. it has shape (m, bs, ...)
         col_major: whether A is in column major or row-major representation
         mode: 'lower', 'upper' or ''
    Returns:
        if mode == 'hermitian': the product A@b in block form, i.e. shape (m, bs, ...)
        if mode == 'lower': the product L@b in block form, i.e. shape (m, bs, ...)
        if mode == 'upper': the product L.T.conj()@b in block form, i.e. shape (m, bs, ...)
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=None, in_axes=(0, None))
    def __block_matmul(L_blocks, X_blocks):
        # compute  L @ X, L.H @ X or A @ X  where A = L + L^H - (I * L)
        # see below

        assert L_blocks.dtype == X_blocks.dtype

        m, bs, *_ = X_blocks.shape
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
        pad_mask_padded[-npad:] = False
        # ij: index of the blocks on the current device; mask: wether they are padding or not
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]
        pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]

        i, j = ij.T
        y = jnp.zeros_like(X_blocks)
        # the gather will take care of summing the correct blocks of each row of the result

        # lower
        if mode == "lower":
            # L @ X
            # assumes blocks on the diagonal are lower triangular
            y = y.at[i].add(jnp.einsum("aij,aj...,a->ai...", L_blocks, X_blocks.at[j].get(), pad_mask.astype(L_blocks.dtype)))
            # jax.debug.print('{d} i:{i} j:{j} m:{m} L:{L} X:{X} y: {y}', d=d, i=i,j=j, m=pad_mask.astype(int), L=L_blocks.ravel(), X=X_blocks.at[j].get().ravel(), y=y.ravel())
        elif mode == "upper":
            # L^H @ X
            # assumes blocks on the diagonal are lower triangular
            y = y.at[j].add(jnp.einsum("aji,aj...,a->ai...", L_blocks.conj(), X_blocks.at[i].get(), pad_mask.astype(L_blocks.dtype)))
        elif mode == "hermitian":
            # equivalent of tril(L) @ X + tril(L, -1)^H @ X
            # assumes blocks on the diagonal are fully populated (i.e. are hermitian)
            offdiag_mask = i != j
            mask = pad_mask & offdiag_mask
            y = y.at[i].add(jnp.einsum("aij,aj...,a->ai...", L_blocks, X_blocks.at[j].get(), pad_mask.astype(L_blocks.dtype)))
            y = y.at[j].add(jnp.einsum("aji,aj...,a->ai...", L_blocks.conj(), X_blocks.at[i].get(), (mask).astype(L_blocks.dtype)))
        else:
            raise ValueError("invalid mode", mode)

        return jax.lax.psum(y, axis_name)

    return __block_matmul(M, b)


block_matmul_hermitian = partial(_block_matmul, mode="hermitian")
block_matmul_tril_forward = partial(_block_matmul, mode="lower")
block_matmul_tril_backward = partial(_block_matmul, mode="upper")


@partial(jax.jit, static_argnames=("col_major", "axis_name", "mode"))
def block_diagonal_trafo(M, t, col_major=False, axis_name="i", mode="left", op=None):
    """symmetric Block-by-block transformation

    Args:
         M: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A or lower triangular matrix L, of mxm blocks of size bs
         t: dense vector/matrix where the first axis is in blocks, i.e. it has shape (m, bs, ...)
         col_major: whether A is in column major or row-major representation
         mode: 'left', 'right' or 'both'
         op: a callable op(block, l, r, mask) wrapped in a jax.tree_util.Partial which returns a new transformed block
            where l (r) contains the elements of t corresponding to the row (column) of the block and mask indicates wether the block is padding
            (true means it is not)
    Returns:
        if op is None:
            if mode == 'left': block_diag(t)^H @ L
            if mode == 'right': L @ block_diag(t)
            if mode == 'both': block_diag(t)^H @ A @ block_diag(t)
        else:
            apply the operation op to each block of the matrix
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=0, in_axes=(0, None))
    def _block_diagonal_trafo(L_blocks, t_blocks):
        # op: Partial, representing custom operation
        # if op is None (default):
        #   multiplication with a block-diagonal matrix from left and/or right
        #   diag(t)^H @ L, diag(t)^H @ L @ diag(t), L @ diag(t)
        # otherwise this calls op(L, ti.conj(), tj, is_padding) on each diagonal block

        assert L_blocks.dtype == t_blocks.dtype

        m, bs, *_ = t_blocks.shape
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        # ij: index of the blocks on the current device
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]

        i, j = ij.T
        if op is None:
            if mode == "left":
                # diag(t)^H @ L
                return jnp.einsum("aij,aik->akj", L_blocks, t_blocks[i].conj())
            elif mode == "right":
                # L @ diag(t)
                return jnp.einsum("aij,ajl->ail", L_blocks, t_blocks[j])
            elif mode == "both":
                # diag(t)^H @ H @ diag(t)
                return jnp.einsum("aij,aik,ajl->akl", L_blocks, t_blocks[i].conj(), t_blocks[j])
            else:
                raise ValueError("invalid mode")
        else:
            pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
            pad_mask_padded[-npad:] = False
            pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]
            return jax.vmap(op)(L_blocks, t_blocks[i].conj(), t_blocks[j], pad_mask)

    return _block_diagonal_trafo(M, t)


@partial(jax.jit, static_argnames=("col_major", "axis_name", "m"))
def extract_diagonal(M, m, f=None, col_major=False, axis_name="i"):
    """extract the blocks on the diagonal

    Args:
         M: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A or lower triangular matrix L, of mxm blocks of size bs
         m: number of blocks in each row an column
         f: a callable wrapped in a jax.tree_util.Partial (optional)
         col_major: whether M is in column major or row-major representation
    Returns:
        the blocks on the diagonal of M
        if f is not None: f is applied to every block before extraction
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=None, in_axes=(0, None))
    def _extract_diagonal(L_blocks, f):
        # extract the diagonal blocks;

        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
        pad_mask_padded[-npad:] = False

        i, j = ij_padded.T
        # max number of diagonal entries, of any device
        maxdiag = ((i == j) & pad_mask_padded).reshape(-1, ndevices).sum(axis=0).max().tolist()

        # ij: index of the blocks on the current device; mask: wether they are padding or not
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]
        pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]

        i, j = ij.T

        diag_mask = (i == j) & pad_mask
        (k,) = jnp.where(diag_mask, size=maxdiag, fill_value=-1)
        i = i[k]
        mask = k >= 0
        L_diag_blocks = L_blocks[k]

        if f is not None:
            L_diag_blocks = jax.vmap(f)(L_diag_blocks)

        def _sum(mask, i, L):
            mask = jnp.expand_dims(mask, tuple(range(mask.ndim, L.ndim)))
            L = jnp.where(mask, L, jnp.zeros_like(L))
            res = jnp.zeros((m, *L.shape[1:]), dtype=L.dtype).at[i].add(L)
            return jax.lax.psum(res, axis_name)

        return jax.tree.map(partial(_sum, mask, i), L_diag_blocks)

    return _extract_diagonal(M, f)


@partial(jax.jit, static_argnames=("col_major", "axis_name", "m"))
def add_diagonal(M, v, m, col_major=False, axis_name="i"):
    """Add a vector to the diagonal

    Args:
         M: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A or lower triangular matrix L, of mxm blocks of size bs
         b: dense vector where the first axis is in blocks, i.e. it has shape (m, bs)
         m: number of blocks in each row an column
         col_major: whether M is in column major or row-major representation
    Returns:
        the equivalent of M + diag(v) in the same representation of M
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=0, in_axes=(0, None))
    def _add_diagonal(L_blocks, diag_vec_blocks):
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
        pad_mask_padded[-npad:] = False

        i, j = ij_padded.T
        # max number of diagonal entries, of any device
        maxdiag = ((i == j) & pad_mask_padded).reshape(-1, ndevices).sum(axis=0).max().tolist()

        # ij: index of the blocks on the current device; mask: wether they are padding or not
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]
        pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]

        i, j = ij.T

        diag_mask = (i == j) & pad_mask
        (k,) = jnp.where(diag_mask, size=maxdiag, fill_value=-1)
        i = i[k]
        mask = k >= 0
        vi = diag_vec_blocks[i]
        vi = jnp.where(mask[:, None], vi, jnp.zeros_like(vi))
        return L_blocks.at[k].add(jax.vmap(jnp.diag)(vi))

    return _add_diagonal(M, v)


@partial(jax.jit, static_argnames=("col_major", "axis_name", "m", "axis"))
def sum_herm(A, m, col_major=False, axis_name="i", axis=None):
    """sum over rows and/or columns

    Args:
         A: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A of mxm blocks of size bs
         m: number of blocks in each row an column
         col_major: whether M is in column major or row-major representation
         axis: 0, 1, or (0,1)
    Returns:
        sum of A over one or both of the axes;
        the axes not summed are blocked in the result, i.e. it has shape (m, bs)
    """

    # sum rows / cols of the hermitian matrix represented by L
    @partial(jax.smap, axis_name=axis_name, out_axes=None, in_axes=(0))
    def _sum_herm(L_blocks):
        _, _, bs = L_blocks.shape
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        ij_unpadded, npad, _, _ = prep_data(m, ndevices, col_major=col_major)
        ij_padded = np.pad(ij_unpadded, ((0, npad), (0, 0)))
        pad_mask_padded = np.ones(len(ij_padded), dtype=bool)
        pad_mask_padded[-npad:] = False

        i, j = ij_padded.T
        # max number of diagonal entries, of any device
        maxdiag = ((i == j) & pad_mask_padded).reshape(-1, ndevices).sum(axis=0).max().tolist()

        # ij: index of the blocks on the current device; mask: wether they are padding or not
        ij = jnp.array(ij_padded.reshape(-1, ndevices, 2))[:, d]
        pad_mask = jnp.array(pad_mask_padded.reshape(-1, ndevices))[:, d]

        i, j = ij.T

        diag_mask = (i == j) & pad_mask

        # sum the diag blocks only with the rows, leave them out with the blocks
        row_sum_blocks = jnp.einsum("aij->ai", L_blocks)
        row_sum = jax.lax.psum(jnp.zeros((m, bs), dtype=L_blocks.dtype).at[i].add(row_sum_blocks), axis_name)
        col_sum_blocks = jnp.einsum("aij,a->aj", L_blocks, ~diag_mask)
        col_sum = jax.lax.psum(jnp.zeros((m, bs), dtype=L_blocks.dtype).at[j].add(col_sum_blocks), axis_name)

        if axis == 0:
            return row_sum.conj() + col_sum
        elif axis == 1:
            return row_sum + col_sum.conj()
        elif axis in ((0, 1), None):
            return (row_sum + col_sum.conj()).sum()
        else:
            raise ValueError(f"invalid axis: {axis}")

    return _sum_herm(A)
