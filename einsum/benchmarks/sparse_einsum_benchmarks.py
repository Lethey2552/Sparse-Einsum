import numpy as np
import sparse
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.utilities.helper_functions import compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer


if __name__ == "__main__":
    print_results = False

    einsum_notation = "abcik,abcjr,abcrk->abcij"

    A = sparse.random((20, 11, 10, 4, 2), idx_dtype=int)
    B = sparse.random((20, 11, 10, 2, 3), idx_dtype=int)
    C = sparse.random((20, 11, 10, 3, 2), idx_dtype=int)
    sparse_arrays = [A, B, C]

    A = sparse.asnumpy(A)
    B = sparse.asnumpy(B)
    C = sparse.asnumpy(C)

    A_coo = Coo_matrix.coo_from_standard(A)
    B_coo = Coo_matrix.coo_from_standard(B)
    C_coo = Coo_matrix.coo_from_standard(C)
    arrays = [A_coo, B_coo, C_coo]

    # Python
    tic = timer()
    try:
        python_res = A @ B @ C
    except:
        python_res = None
    toc = timer()

    python_time = toc - tic

    # Sparse
    tic = timer()
    sparse_res = sparse.einsum(einsum_notation, sparse_arrays[0], sparse_arrays[1], sparse_arrays[2])
    toc = timer()

    sparse_time = toc - tic

    # Sparse Einsum
    tic = timer()
    sparse_einsum_res = sparse_einsum(einsum_notation, arrays)
    toc = timer()

    sparse_einsum_time = toc - tic

    print(f"Shapes: Python - {python_res.shape if python_res is not None else 'None'},    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"Results are correct: {compare_matrices(sparse_einsum_res, sparse.asnumpy(sparse_res))}")

    print(f"Python time: {python_time}s")
    print(f"Sparse time: {sparse_time}s")
    print(f"Sparse Einsum time: {sparse_einsum_time}s")

    if print_results:
        print(f"Python result:\n{python_res}")
        print(f"Sparse result:\n{sparse.asnumpy(sparse_res)}\n")
        print(f"Sparse Einsum result:\n{sparse_einsum_res.coo_to_standard()}")