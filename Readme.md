Distributed solver library written in pure jax

### Features:
- support for lower-triangular and symmetric/hermitian matrices
- cholesky decomposition
- triangular solve
- matrix-vector multiplication
- direct construction of the matrix in the internal 2D block cyclic storage format

### Stroage Format
The library is based on a 2D block cyclic storage format, storing only the lower triangular part of a given matrix as explained in the following example.
Given a hermitian 3x3 block matrix, with each block of a the same block size

```
[A B̄ D̄]
[B C Ē]
[D E F]
```
we disribute the lower triangular blocks (A,B,C,D,E,F) on 3 jax devices (0,1,2) in one of two possible orderings (the remaining blocks are redundant because of the hermitianity assumption):
row-major format:
```
[0 X X]
[1 2 X]
[0 1 2]
```

column-major format:
```
[0 X X]
[1 0 X]
[2 1 2]
```
The the matrix is then stored on the devices as (assuming column-major ordering)
```
0: [A, C]
1: [B, E]
2: [D, F]
```
The data can be represented as one single array of blocks `[A, C, B, E, D, F]`, sharded along axis 0. In cases where the number of blocks on each device is not uniform we pad with zero blocks.

### Usage
see [examples/kernel_ridge_regression.py](examples/kernel_ridge_regression.py)
