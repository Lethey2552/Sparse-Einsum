import numpy as np
import sparse
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.utilities.helper_functions import compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer

if __name__ == "__main__":
    print_results = False

    einsum_notation = "tbacik,abcrk,bacjr->abcij"

    A = sparse.random((5, 11, 40, 10, 4, 2), density=0.2, idx_dtype=int)
    B = sparse.random((40, 11, 10, 3, 2), density=0.1, idx_dtype=int)
    C = sparse.random((11, 40, 10, 7, 3), density=0.1, idx_dtype=int)
    sparse_arrays = [A, B, C]
    
    A = sparse.asnumpy(A)
    B = sparse.asnumpy(B)
    C = sparse.asnumpy(C)

    A_coo = Coo_matrix.coo_from_standard(A)
    B_coo = Coo_matrix.coo_from_standard(B)
    C_coo = Coo_matrix.coo_from_standard(C)
    arrays = [A_coo, B_coo, C_coo]

    # Numpy Dense
    tic = timer()
    numpy_res = np.einsum(einsum_notation, A, B, C)
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

    print(f"Shapes: Numpy - {numpy_res.shape if numpy_res is not None else 'None'},    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"""Results are correct:\n    Sparse Einsum - Sparse: {compare_matrices(sparse_einsum_res, sparse.asnumpy(sparse_res))}
    Sparse Einsum - Numpy: {compare_matrices(sparse_einsum_res, numpy_res)}""")

    print(f"Numpy time: {python_time}s")
    print(f"Sparse time: {sparse_time}s")
    print(f"Sparse Einsum time: {sparse_einsum_time}s")

    if print_results:
        print(f"Numpy result:\n{numpy_res}")
        print(f"Sparse result:\n{sparse.asnumpy(sparse_res)}\n")
        print(f"Sparse Einsum result:\n{sparse_einsum_res.coo_to_standard()}")