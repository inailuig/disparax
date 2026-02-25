import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
import numpy as np
from functools import partial, lru_cache

from .data import prep_data


@partial(jax.jit, static_argnames=("col_major", "axis_name"))
def block_forward_solve(L, b, col_major=False, axis_name="i"):
    """solve the triangular system L x = b

    Args:
        L: lower-triangular 2d block-cyclic representation of a lower-trianuglar matrix L of mxm blocks of size bs
        b: dense vector/matrix where the first axis is in blocks, i.e. it has shape (m, bs, ...)
        col_major: whether M is in column major or row-major representation
    Returns:
        the solution x of the linear system of equations L x = b
        where x is stored with first axis in blocks, i.e. shape (m, bs, ...)
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=None, in_axes=(0, None))
    def _block_forward_solve(L_blocks, b_blocks):
        # solve L y = b
        m, bs = b_blocks.shape
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        _, _, row_data, _ = prep_data(m, ndevices, col_major=col_major)
        dev_row_ind_diag, dev_row_ind_offdiag_data, dev_row_ind_offdiag_col, dev_ind_diag = row_data

        def body_fun(row, y_blocks):
            diag_ind = jnp.array(dev_row_ind_diag)[d, row]
            blk_ind = jnp.array(dev_row_ind_offdiag_data)[d, row]
            col_ind = jnp.array(dev_row_ind_offdiag_col)[d, row]
            mask = jnp.array(dev_row_ind_offdiag_col)[d, row] >= 0
            diag_mask = jnp.array(dev_ind_diag)[row] == d

            tmp = jnp.einsum("aij,a,aj->i", L_blocks[blk_ind], mask, y_blocks[col_ind])
            rhs = b_blocks[row] - jax.lax.psum(tmp, axis_name)
            tmp = jax.lax.linalg.triangular_solve(L_blocks[diag_ind], rhs, lower=True, transpose_a=True)

            # TODO send from the one which has the diag element to everybody else, instead of psum
            tmp = jax.lax.psum(jax.lax.select(diag_mask, tmp, jnp.zeros_like(tmp)), axis_name)
            return y_blocks.at[row].set(tmp)

        init_val = jnp.zeros_like(b_blocks)
        return jax.lax.fori_loop(0, m, body_fun, init_val)

    return _block_forward_solve(L, b)


@partial(jax.jit, static_argnames=("col_major", "axis_name"))
def block_backward_solve(L_blocks, b_blocks, col_major=False, axis_name="i"):
    """solve the triangular system L^H y = b

    Args:
            L: lower-triangular 2d block-cyclic representation of a lower-trianuglar matrix L of mxm blocks of size bs
            b: dense vector/matrix where the first axis is in blocks, i.e. it has shape (m, bs, ...)
            col_major: whether M is in column major or row-major representation
    Returns:
        the solution x of the linear system of equations L^H y = b
        where x is stored with first axis in blocks, i.e. shape (m, bs, ...)
    """

    @partial(jax.smap, axis_name=axis_name, out_axes=None, in_axes=(0, None))
    def _block_backward_solve(L_blocks, b_blocks):
        #
        m, bs = b_blocks.shape
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)

        _, _, _, col_data = prep_data(m, ndevices, col_major=col_major)
        dev_col_ind_diag, dev_col_ind_offdiag, dev_offdiag, devs_diag = col_data

        def body_fun(k, y_blocks):
            col = m - k - 1  # right to left
            diag_ind = jnp.array(dev_col_ind_diag)[d, col]
            blk_ind = jnp.array(dev_col_ind_offdiag)[d, col]
            col_ind = jnp.array(dev_offdiag)[d, col]
            mask = jnp.array(dev_offdiag)[d, col] >= 0
            diag_mask = jnp.array(devs_diag)[col] == d

            tmp = jnp.einsum("aij,a,ai->j", L_blocks[blk_ind].conj(), mask, y_blocks[col_ind])
            rhs = b_blocks[col] - jax.lax.psum(tmp, axis_name)
            tmp = jax.lax.linalg.triangular_solve(L_blocks[diag_ind], rhs, lower=True, transpose_a=False, conjugate_a=True)

            # TODO send from the one which has the diag element to everybody else, instead of psum
            tmp = jax.lax.psum(jax.lax.select(diag_mask, tmp, jnp.zeros_like(tmp)), axis_name)
            return y_blocks.at[col].set(tmp)

        init_val = jnp.zeros_like(b_blocks)
        return jax.lax.fori_loop(0, m, body_fun, init_val)

    return _block_backward_solve(L_blocks, b_blocks)
