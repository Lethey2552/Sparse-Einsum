import numpy as np
import sparse
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.utilities.helper_functions import compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer


if __name__ == "__main__":
    einsum_notation = "abcik,backj->abcij"

    A = sparse.random((4, 11, 10, 3, 2), idx_dtype=int)
    B = sparse.random((11, 4, 10, 2, 3), idx_dtype=int)
    sparse_arrays = [A, B]

    A = sparse.asnumpy(A)
    B = sparse.asnumpy(B)

    A_coo = Coo_matrix.coo_from_standard(A)
    B_coo = Coo_matrix.coo_from_standard(B)
    arrays = [A_coo, B_coo]

    # Python
    tic = timer()
    # python_res = A @ B
    toc = timer()

    python_time = toc - tic

    # Sparse
    tic = timer()
    sparse_res = sparse.einsum(einsum_notation, sparse_arrays[0], sparse_arrays[1])
    toc = timer()

    sparse_time = toc - tic

    # Sparse Einsum
    tic = timer()
    sparse_einsum_res = sparse_einsum(einsum_notation, arrays)
    toc = timer()

    sparse_einsum_time = toc - tic

    print(f"Shapes: Python - python_res.shape,    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"Results are correct: {compare_matrices(sparse_einsum_res, sparse.asnumpy(sparse_res))}")

    print(f"Python result: {python_time}s")
    print(f"Sparse result: {sparse_time}s")
    print(f"Sparse Einsum result: {sparse_einsum_time}s")