import numpy as np
import opt_einsum as oe
import sparse
import sqlite3 as sql
import torch
from einsum.experiments.util import delete_values_to_match_density
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, sql_einsum_execute)
from einsum.utilities.classes.coo_matrix import Coo_tensor
from timeit import default_timer as timer

if __name__ == "__main__":
    print_results = False
    run_np = False
    run_sql_einsum = False
    run_torch = False

    einsum_notation = "xtbacik,ysabcrk,ubacjr->abcji"

    A = sparse.random((20, 25, 80, 70, 10, 4, 6),
                      density=0.001, idx_dtype=int)
    B = sparse.random((5, 30, 70, 80, 10, 3, 6),
                      density=0.01, idx_dtype=int)
    C = sparse.random((40, 80, 70, 10, 7, 3),
                      density=0.01, idx_dtype=int)

    # einsum_notation = "bfi,bfr,fbj,fkj,bri,fbk,rfj->bij"

    # A = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # B = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # C = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # D = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # E = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # F = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # G = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)

    # einsum_notation = "ij,iml,lo,jk,kmn,no->"

    # A = sparse.random((2, 3), density=1.0, idx_dtype=int, random_state=0)
    # B = sparse.random((2, 4, 5), density=1.0, idx_dtype=int, random_state=1)
    # C = sparse.random((5, 2), density=1.0, idx_dtype=int, random_state=2)
    # D = sparse.random((3, 7), density=1.0, idx_dtype=int, random_state=3)
    # E = sparse.random((7, 4, 2), density=1.0, idx_dtype=int, random_state=4)
    # F = sparse.random((2, 2), density=1.0, idx_dtype=int, random_state=5)

    sparse_arrays = [A, B, C]
    dense_arrays = []
    torch_arrays = []
    sparse_einsum_arrays = []
    sparse_einsum_arrays_legacy = []

    for i in sparse_arrays:
        dense_array = sparse.asnumpy(i)
        dense_arrays.append(dense_array)
        torch_arrays.append(torch.from_numpy(dense_array))
        sparse_einsum_arrays.append(Coo_tensor.from_numpy(dense_array))
        sparse_einsum_arrays_legacy.append(Coo_tensor.from_numpy(dense_array))

    if run_np:
        # Numpy Dense
        tic = timer()
        numpy_res = np.einsum(einsum_notation, *dense_arrays)
        toc = timer()
        numpy_time = toc - tic

    if run_torch:
        tic = timer()
        torch_res = oe.contract(
            einsum_notation, *torch_arrays, backend="torch")
        toc = timer()
        torch_time = toc - tic

    # Sparse
    tic = timer()
    sparse_res = sparse.einsum(
        einsum_notation, *sparse_arrays)
    toc = timer()

    sparse_time = toc - tic

    # Sparse Einsum
    tic = timer()
    sparse_einsum_res = sparse_einsum(
        einsum_notation, sparse_einsum_arrays, show_progress=False, allow_alter_input=True)
    toc = timer()

    sparse_einsum_time = toc - tic

    # Sparse Einsum
    tic = timer()
    sparse_einsum_legacy_res = sparse_einsum(
        einsum_notation, sparse_einsum_arrays_legacy, show_progress=False, allow_alter_input=True, no_parallelization=True)
    toc = timer()

    sparse_einsum_time_legacy = toc - tic

    print(
        f"Shapes: Numpy - {numpy_res.shape if run_np else 'None'},    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"""Results are correct:\n    Sparse Einsum - Sparse: {np.isclose(np.sum(sparse_res), np.sum(sparse_einsum_res))}
    Sparse Einsum - Numpy: {np.isclose(np.sum(numpy_res), np.sum(sparse_einsum_res)) if run_np else 'None'}""")

    print(f"Numpy time: {numpy_time if run_np else 'None'}s")
    print(f"Torch time: {torch_time if run_torch else 'None'}s")
    print(f"Sparse time: {sparse_time}s")
    print(f"Sparse Einsum time: {sparse_einsum_time}s")
    print(f"Sparse Einsum Legacy time: {sparse_einsum_time_legacy}s")

    if print_results:
        print(f"\nNumpy result:\n{numpy_res if run_np else 'None'}")
        print(f"Sparse result:\n{sparse.asnumpy(sparse_res)}")
        print(f"Sparse Einsum result:\n{sparse_einsum_res}")

    if run_sql_einsum:
        tensor_names = []
        tensor_dict = {}
        for i, arr in enumerate(dense_arrays):
            tensor_dict["T" + str(i)] = arr
            tensor_names.append("T" + str(i))

        query, res_shape = sql_einsum_query(
            einsum_notation, tensor_names, tensor_dict)

        tic = timer()
        result = sql_einsum_execute(query, res_shape)
        toc = timer()

        print("sum[OUTPUT]:", np.sum(result),
              np.sum(sparse_einsum_res))
        print(f"SQL Einsum time: {toc - tic}s")
