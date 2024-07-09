import numpy as np
import sparse
import torch
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.utilities.helper_functions import compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer

if __name__ == "__main__":
    print_results = False
    run_np = False

    # einsum_notation = "tbacik,sabcrk,ubacjr->abcij"

    # A = sparse.random((25, 80, 70, 10, 4, 2),
    #                   density=0.01, idx_dtype=int)
    # B = sparse.random((30, 70, 80, 10, 3, 2),
    #                   density=0.01, idx_dtype=int)
    # C = sparse.random((40, 80, 70, 10, 7, 3),
    #                   density=0.01, idx_dtype=int)

    # einsum_notation = "abcik,abckr,abcrj->abcij"

    # A = sparse.random((2, 2, 3, 2, 2), density=0.5, idx_dtype=int)
    # B = sparse.random((2, 2, 3, 2, 2), density=0.5, idx_dtype=int)
    # C = sparse.random((2, 2, 3, 2, 2), density=0.5, idx_dtype=int)

    einsum_notation = "ab,bc->ac"

    A = sparse.random((2, 2), density=1.0, idx_dtype=int)
    B = sparse.random((2, 2), density=1.0, idx_dtype=int)

    sparse_arrays = [A, B]
    dense_arrays = []

    for i in sparse_arrays:
        dense_arrays.append(sparse.asnumpy(i))
        print("INPUT:")
        print(dense_arrays[-1])
        print()

    if run_np:
        # Numpy Dense
        tic = timer()
        numpy_res = np.einsum(einsum_notation, *dense_arrays)
        toc = timer()

        numpy_time = toc - tic

    # Sparse
    tic = timer()
    sparse_res = sparse.einsum(
        einsum_notation, *sparse_arrays)
    toc = timer()

    sparse_time = toc - tic

    # Sparse Einsum
    tic = timer()
    sparse_einsum_res = sparse_einsum(einsum_notation, dense_arrays)
    toc = timer()

    sparse_einsum_time = toc - tic

    print(
        f"Shapes: Numpy - {numpy_res.shape if run_np else 'None'},    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"""Results are correct:\n    Sparse Einsum - Sparse: {compare_matrices(sparse_einsum_res, sparse.asnumpy(sparse_res))}
    Sparse Einsum - Numpy: {compare_matrices(sparse_einsum_res, numpy_res) if run_np else 'None'}""")

    print(f"Numpy time: {numpy_time if run_np else 'None'}s")
    print(f"Sparse time: {sparse_time}s")
    print(f"Sparse Einsum time: {sparse_einsum_time}s")

    if print_results:
        print(f"Numpy result:\n{numpy_res if run_np else 'None'}")
        print(f"Sparse result:\n{sparse.asnumpy(sparse_res)}\n")
        print(f"Sparse Einsum result:\n{sparse_einsum_res.coo_to_standard()}")

    torch_array = [torch.from_numpy(i) for i in dense_arrays]

    tic = timer()
    torch_res = torch.einsum(einsum_notation, *torch_array)
    toc = timer()

    torch_time = toc - tic

    print(f"Torch time: {torch_time}s")
