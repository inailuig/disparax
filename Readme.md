Distributed solver library written in pure jax

### Features:
- support for lower-triangular and symmetric/hermitian matrices
- cholesky decomposition
- triangular solve
- matrix-vector multiplication
- direct construction of the matrix in the internal 2D block cyclic storage format

### Stroage Format
The library is based on a lower-triangular 2D block cyclic storage format.
We only store the lower triangular part of a block matrix as explained in the following example:

Given a hermitian `n x n` matrix, partitioned into a `3 x 3` grid of blocks of size `bs x bs`, where we assume the blocks size divides the size of the matrix such that `n = 3*bs`,

```
[A B̄ D̄]
[B C Ē]
[D E F]
```
We disribute the lower triangular blocks (A,B,C,D,E,F) on 3 jax devices (0,1,2) in one of two possible orderings:

```
row-major:       column-major:
[0 X X]          [0 X X]
[1 2 X]          [1 0 X]
[0 1 2]          [2 1 2]
```
Hree the remaining blocks marked by `X` are redundant because of the hermitianity assumption, and thus we do not store them.

The the matrix is then stored on the devices as
```
row-major:      column-major:
0: [A, D]       0: [A, C]
1: [B, E]       1: [B, E]
2: [C, F]       2: [D, F]
```
The data can be represented as one single array of blocks `row-major:[A, D, B, E, C, F]`/`column-major:[A, C, B, E, D, F]`, of size `(n_blocks, bs, bs)` sharded along axis 0. 
In cases where the number of blocks on each device is not uniform we pad with zero blocks such that n_blocks is becomes a multiple of it.

### Installation
```bash
pip install git+https://github.com/inailuig/disparax.git
```

### Usage
see [examples/kernel_ridge_regression.py](examples/kernel_ridge_regression.py)
