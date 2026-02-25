import pytest

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial
from jax.sharding import Mesh, AxisType, set_mesh, reshard
from jax import P

from disparax import *

mesh = Mesh(jax.devices(), "i", axis_types=(AxisType.Explicit))
set_mesh(mesh)


# TODO turn this into fixture
def setup():
    n = 128
    bs = 32
    k = 11
    m = n // bs

    A = np.random.uniform(size=(n, n)) + 1.0j * np.random.uniform(size=(n, n)) + np.eye(n)
    A = 0.5 * (A + A.T.conj()) + np.eye(n) * 10
    b = np.random.uniform(size=(n,)) + 1.0j * np.random.uniform(size=(n,))
    X = np.random.uniform(size=(n, k)) + 1.0j * np.random.uniform(size=(n, k)) + 1.0j * np.random.uniform(size=(n, k)) + 1.0j * np.random.uniform(size=(n, k))

    L = np.linalg.cholesky(A)
    # np.testing.assert_allclose(np.einsum('ij,kj->ik', L, L.conj()), A)
    return n, bs, k, m, A, b, X, L


@pytest.mark.parametrize("col_major", [True, False])
def test_solve(col_major):
    n, bs, k, m, A, b, X, L = setup()
    L_block_data = pack_blocks(L, bs, jax.device_count(), col_major=col_major)
    b_block = b.reshape(-1, bs)
    np.testing.assert_allclose(block_forward_solve(L_block_data, b_block, col_major=col_major).ravel(), np.linalg.inv(L) @ b)
    np.testing.assert_allclose(block_backward_solve(L_block_data, b_block, col_major=col_major).ravel(), np.linalg.inv(L.T.conj()) @ b)


def test_cholesky(col_major=True):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    L_block_data = pack_blocks(L, bs, jax.device_count(), col_major=col_major)
    expected = reshard(L_block_data, P()).reshape(jax.device_count(), -1)
    L2_block_data = block_cholesky(A_block_data, m, col_major=col_major)
    got = reshard(L2_block_data, P()).reshape(jax.device_count(), -1)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("col_major", [True, False])
def test_add_diagonal(col_major):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    d = np.random.uniform(size=(n,))
    expected = pack_blocks(A + jnp.diag(d), bs, jax.device_count(), col_major=col_major)
    got = add_diagonal(A_block_data, d.reshape(m, bs), m, col_major=col_major)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("col_major", [True, False])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
def test_sum_herm(col_major, axis):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    np.testing.assert_allclose(sum_herm(A_block_data, m, col_major=col_major, axis=axis).ravel(), A.sum(axis=axis))


@pytest.mark.parametrize("col_major", [True, False])
def test_matmul(col_major):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    L_block_data = pack_blocks(L, bs, jax.device_count(), col_major=col_major)
    X_block = X.reshape(m, bs, -1)
    y_block1 = block_matmul_tril_backward(L_block_data, X_block, col_major=col_major)
    np.testing.assert_allclose(y_block1, (L.T.conj() @ X).reshape(y_block1.shape))
    y_block2 = block_matmul_tril_forward(L_block_data, X_block, col_major=col_major)
    np.testing.assert_allclose(y_block2, (L @ X).reshape(y_block2.shape))
    y_block3 = block_matmul_hermitian(A_block_data, X_block, col_major=col_major)
    np.testing.assert_allclose(y_block3, (A @ X).reshape(y_block3.shape))


@pytest.mark.parametrize("col_major", [True, False])
def test_compute_blocks(col_major):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    f = Partial(lambda A, i, j: A[i, :][:, j], A)
    x = jnp.arange(n)
    A_block_data2 = compute_blocks(f, x, bs=bs, col_major=col_major)
    np.testing.assert_allclose(A_block_data2, A_block_data)


def test_solve(col_major=True):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    b_blocks = b.reshape(m, bs)
    expected = jnp.linalg.solve(A, b)
    got = solve(A_block_data, b_blocks, m, col_major=col_major).ravel()
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("col_major", [True, False])
def test_extract_diagonal(col_major):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    f = jax.vmap(jax.lax.sin)
    expected = f(A.reshape(m, bs, m, bs).transpose(0, 2, 1, 3).diagonal(axis1=0, axis2=1).transpose(2, 0, 1))
    got = extract_diagonal(A_block_data, m, Partial(f), col_major=col_major)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mode", ["both", "left", "right", "op"])
@pytest.mark.parametrize("col_major", [True, False])
def test_block_diagonal_trafo(col_major, mode):
    n, bs, k, m, A, b, X, L = setup()
    A_block_data = pack_blocks(A, bs, jax.device_count(), col_major=col_major)
    L_block_data = pack_blocks(L, bs, jax.device_count(), col_major=col_major)
    t_blocks = jnp.linalg.qr(np.random.normal(size=(m, bs, bs)) + 1.0j * np.random.normal(size=(m, bs, bs)))[0]

    if mode == "both":
        t_blocks_nonsquare = t_blocks[:, :, : bs // 2]
        B = jnp.einsum("aibj,aik,bjl->akbl", A.reshape(m, bs, m, bs), t_blocks_nonsquare.conj(), t_blocks_nonsquare).reshape(m * bs // 2, m * bs // 2)
        expected = pack_blocks(B, bs // 2, jax.device_count(), col_major=col_major)
        got = block_diagonal_trafo(A_block_data, t_blocks_nonsquare, col_major=col_major, mode="both")
        np.testing.assert_allclose(got, expected)
    elif mode == "left":
        B = jnp.einsum("aibj,aik->akbj", L.reshape(m, bs, m, bs), t_blocks.conj()).reshape(L.shape)
        expected = pack_blocks(B, bs, jax.device_count(), col_major=col_major)
        got = block_diagonal_trafo(L_block_data, t_blocks, col_major=col_major, mode="left")
        np.testing.assert_allclose(got, expected)
    elif mode == "right":
        B = jnp.einsum("aibj,bjl->aibl", L.reshape(m, bs, m, bs), t_blocks).reshape(L.shape)
        expected = pack_blocks(B, bs, jax.device_count(), col_major=col_major)
        got = block_diagonal_trafo(L_block_data, t_blocks, col_major=col_major, mode="right")
        np.testing.assert_allclose(got, expected)
    elif mode == "op":
        u = np.random.normal(size=n) + 1.0j * np.random.normal(size=n)
        c = np.random.uniform()

        def _op(c, A, l, r, mask):
            return A + mask * (-l[:, None] - r[None, :] + c)

        B = A - u[:, None].conj() - u[None, :] + c
        expected = pack_blocks(B, bs, jax.device_count(), col_major=col_major)
        got = block_diagonal_trafo(A_block_data, u.reshape(m, bs), col_major=col_major, op=Partial(_op, c))
        np.testing.assert_allclose(got, expected)
    else:
        raise ValueError("unknown test mode")
