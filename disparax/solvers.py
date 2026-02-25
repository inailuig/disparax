from .triangular_solve import block_forward_solve, block_backward_solve
from .cholesky import block_cholesky


def solve(A, b, m, col_major=False, axis_name="i"):
    """solve a linear system using cholesky decomposition and triangular solve

    Args:
            A: lower-triangular 2d block-cyclic representation of a hermitian/symmetric matrix A of mxm blocks of size bs
            b: dense vector/matrix where the first axis is in blocks, i.e. it has shape (m, bs, ...)
            m: number of blocks in each row an column
            col_major: whether M is in column major or row-major representation
            axis: 0, 1, or (0,1)
    Returns:
        the solution x of the linear system of equations A x = b
        where x is stored with first axis in blocks, i.e. shape (m, bs, ...)
    """

    #
    L_block_data = block_cholesky(A, m, col_major=col_major, axis_name=axis_name)
    tmp = block_forward_solve(L_block_data, b, col_major=col_major, axis_name=axis_name)
    return block_backward_solve(L_block_data, tmp, col_major=col_major, axis_name=axis_name)
