import numpy as np
import opt_einsum as oe
import sparse
import sqlite3 as sql
import torch
from einsum.experiments.util import delete_values_to_match_density
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, get_matrix_from_sql_response)
from timeit import default_timer as timer

if __name__ == "__main__":
    print_results = True
    run_np = False
    run_sql_einsum = False
    run_torch = False

    # einsum_notation = "xtbacik,ysabcrk,ubacjr->abcji"

    # A = sparse.random((20, 25, 80, 70, 10, 4, 6),
    #                   density=0.001, idx_dtype=int)
    # B = sparse.random((5, 30, 70, 80, 10, 3, 6),
    #                   density=0.01, idx_dtype=int)
    # C = sparse.random((40, 80, 70, 10, 7, 3),
    #                   density=0.01, idx_dtype=int)

    # einsum_notation = "bfi,bfr,fbj,fkj,bri,fbk,rfj->bij"

    # A = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # B = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # C = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # D = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # E = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # F = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)
    # G = sparse.random((2, 2, 2), density=1.0, idx_dtype=int)

    einsum_notation = "ij,iml,lo,jk,kmn,no->"

    A = sparse.random((2, 3), density=1.0, idx_dtype=int, random_state=0)
    B = sparse.random((2, 4, 5), density=1.0, idx_dtype=int, random_state=1)
    C = sparse.random((5, 2), density=1.0, idx_dtype=int, random_state=2)
    D = sparse.random((3, 7), density=1.0, idx_dtype=int, random_state=3)
    E = sparse.random((7, 4, 2), density=1.0, idx_dtype=int, random_state=4)
    F = sparse.random((2, 2), density=1.0, idx_dtype=int, random_state=5)

    sparse_arrays = [A, B, C, D, E, F]
    dense_arrays = []
    torch_arrays = []

    for i in sparse_arrays:
        dense_array = sparse.asnumpy(i)
        dense_arrays.append(dense_array)
        if run_torch:
            torch_arrays.append(torch.from_numpy(dense_array))

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
    sparse_einsum_res = sparse_einsum(einsum_notation, dense_arrays)
    toc = timer()

    sparse_einsum_time = toc - tic

    print(
        f"Shapes: Numpy - {numpy_res.shape if run_np else 'None'},    Sparse - {sparse_res.shape},    Sparse Einsum - {sparse_einsum_res.shape}")
    print(f"""Results are correct:\n    Sparse Einsum - Sparse: {np.isclose(np.sum(sparse_res), np.sum(sparse_einsum_res))}
    Sparse Einsum - Numpy: {np.isclose(np.sum(numpy_res), np.sum(sparse_einsum_res)) if run_np else 'None'}""")

    print(f"Numpy time: {numpy_time if run_np else 'None'}s")
    print(f"Torch time: {torch_time if run_torch else 'None'}s")
    print(f"Sparse time: {sparse_time}s")
    print(f"Sparse Einsum time: {sparse_einsum_time}s")

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

        query = sql_einsum_query(einsum_notation, tensor_names, tensor_dict)

        db_connection = sql.connect("SQL_einsum.db")
        db = db_connection.cursor()

        tic = timer()
        result = db.execute(query)
        result = get_matrix_from_sql_response(result.fetchall())
        toc = timer()

        print("sum[OUTPUT]:", np.sum(result),
              np.sum(sparse_einsum_res))
        print(f"SQL Einsum time: {toc - tic}s")
