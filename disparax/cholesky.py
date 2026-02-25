import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
import numpy as np
from functools import partial, lru_cache

from .data import prep_data


@partial(jax.jit, static_argnames=("m", "col_major", "axis_name"))
def block_cholesky(A, m, col_major=False, axis_name="i"):
    """Cholesky decomposition
    Args:
         A: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix
         m: number of blocks in each row an column
         col_major: whether A is in column major or row-major representation
    Returns:
        lower-triangular 2d block-cyclic representation of the Cholesky factor of A
    """
    # m = n//bs for a nxn matrix
    assert A.ndim == 3
    if not col_major:
        raise NotImplementedError

    @partial(jax.smap, axis_name=axis_name, out_axes=0, in_axes=0)
    def _block_cholesky(A):
        ndevices = jax.lax.axis_size(axis_name)
        d = jax.lax.axis_index(axis_name)
        bs = A.shape[-1]

        # send data to next device
        # perm = tuple(np.roll(np.arange(ndevices), +1).tolist())

        *_, data_row, data_col = prep_data(m, ndevices, col_major=col_major)
        maxperrow = data_row[2].shape[-1]
        maxpercol = data_col[2].shape[-1]

        G = jnp.zeros_like(A)
        for j in range(m):
            rowj_diag_ind = jnp.array(data_row[0])[d, j]
            rowj_blk_ind = jnp.array(data_row[1])[d, j]
            rowj_col_ind = jnp.array(data_row[2])[d, j]
            rowj_mask = jnp.array(data_row[2])[d, j] >= 0
            diagj_mask = jnp.array(data_row[3])[j] == d

            rowj = G[rowj_blk_ind]
            tmp = jnp.einsum("aik, a, ajk->ij", rowj, rowj_mask, rowj.conj())
            Ajj = jax.lax.select(diagj_mask, A[rowj_diag_ind], jnp.zeros_like(A[rowj_diag_ind]))
            S = jax.lax.psum(Ajj - tmp, axis_name)
            Gjj = jnp.linalg.cholesky(S)
            G = G.at[rowj_diag_ind].set(jax.lax.select(diagj_mask, Gjj, G.at[rowj_diag_ind].get()))

            # TODO compute multiple rows at a time whenever there are more k than devices
            # i.e. send to each other device at most once
            # i.e. turn this into a loop of exactly min(m-j, ndevices) iterations

            num_rows_below = m - j - 1
            maxpercolbelow = -(-num_rows_below // ndevices)
            tmp2_cache = jnp.zeros((num_rows_below, bs, bs), dtype=rowj.dtype)

            for ik in range(min(num_rows_below, ndevices)):
                # TODO if we want to make it a jax loop, change this to always pass it down one row at a time, not send it directly a few cols down like here
                # so that the perm is data independent
                perm = tuple(np.roll(np.arange(ndevices), ik + 1).tolist())
                # this sends the full row; TODO we would only need up to j-1
                rowj_tmp = jax.lax.pshuffle(rowj, axis_name, perm)

                for v in range(maxpercolbelow):
                    ik_ = ik + v * ndevices
                    k = j + 1 + ik_
                    if k < m:
                        rowk_blk_ind = jnp.array(data_row[1])[d, k]
                        rowk = G[rowk_blk_ind]
                        # only up to the col j
                        rowk_mask = (jnp.array(data_row[2])[d, k] >= 0) * (jnp.array(data_row[2])[d, k] < j)
                        tmp2 = jnp.einsum("aik, a, ajk->ij", rowk, rowk_mask, rowj_tmp.conj())
                        tmp2_cache = tmp2_cache.at[ik_].set(tmp2)

            tmp2_cache = jax.lax.psum(tmp2_cache, axis_name)

            for u in range(maxpercolbelow):
                ik = u * ndevices
                k = u * ndevices + j + 1

                # size of what is left
                w = min(ik + ndevices, num_rows_below) - ik

                tmp2 = tmp2_cache[ik : ik + w]

                kj_mask = jnp.array(data_row[2])[d, k : k + w] == j
                have_kj = (jnp.array(data_row[2])[d, k : k + w] == j).any()
                rowk_blk_ind = jnp.array(data_row[1])[d, k : k + w]

                # get the single one where its the right col (might be none, then the ind is just 0 and we filter later)
                kj_ind = jnp.einsum("ai,ai", rowk_blk_ind, kj_mask)
                Akj = A[kj_ind]

                mask2 = kj_mask.sum(axis=-1)
                Sk = Akj - jnp.einsum("aij,a", tmp2, mask2)
                tmp3 = jax.lax.linalg.triangular_solve(Gjj, Sk, lower=True, transpose_a=True, conjugate_a=True)
                G = G.at[kj_ind].set(jax.lax.select(have_kj, tmp3, G.at[kj_ind].get()))
        return G

    return _block_cholesky(A)
