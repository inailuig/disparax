# In this example we show how do use disparax to perform kernel ridge regression
# where the kernel matrix is constructed directly in the lower-triangular 2d block-cyclic representation
# and inverted using the distributed cholesky and triangular solve

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_ENABLE_X64"] = "1"

import matplotlib.pyplot as plt
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.sharding import Mesh, AxisType, set_mesh

import disparax as dpx

mesh = Mesh(jax.devices(), "i", axis_types=(AxisType.Explicit))
set_mesh(mesh)

n_train = 32  # matrix size
block_size = 8  # block size, needs to be a divisor of n_train
X_train = jnp.linspace(-3.0, 3.0, n_train)[:, None]
y_train = jnp.sin(X_train) + 0.1 * jax.random.normal(jax.random.PRNGKey(123), shape=(n_train, 1))


def kernel(x1, x2, sigma=0.5):
    return jnp.exp(-jnp.sum((x1[:, None] - x2[None, :]) ** 2, axis=-1) / (2 * sigma**2))


@partial(jax.jit, static_argnames="block_size")
def train(X, y, block_size, sigma=0.5, lam=1e-3):
    m = len(X) // block_size  # m x m blocks each of size block_size x block_size
    # compute lower triangular part of the symmetric matrix K(X,X)
    # here we construct the triangular 2d block-cyclic representation directly
    # by evaluating the kernel, avoiding to store the full matrix or having to reshard it
    # use column major as cholesky has not yet been implemented for row major
    K = dpx.compute_blocks(Partial(kernel, sigma=sigma), X, block_size, col_major=True)
    # add diagonal shift K = K + lam * I
    K = dpx.add_diagonal(K, jnp.ones((m, block_size)) * lam, m, col_major=True)
    # reshape rhs to match the block structure
    y_blocks = y.reshape(m, block_size)
    # solve K^-1 y
    res = dpx.solve(K, y_blocks, m, col_major=True)
    # flatten result from (m, block_size) to (n_train)
    return res.ravel()


@jax.jit
def predict(X_train, alpha, X_test, sigma=0.5):
    return kernel(X_test, X_train, sigma) @ alpha


sigma = 1.5
lam = 1e-3
alpha = train(X_train, y_train, block_size, sigma, lam)

X_test = jnp.linspace(-3.0, 3.0, 200)[:, None]
y_pred = predict(X_train, alpha, X_test, sigma)

plt.plot(X_test, jnp.sin(X_test), label="sin(x)", linestyle="--")
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_test, y_pred, label="KRR Prediction")
plt.legend()
plt.show()
