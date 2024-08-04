import numpy as np
import opt_einsum as oe
import sparse
import sqlite3 as sql
import torch
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, get_matrix_from_sql_response)
from einsum.utilities.helper_functions import compare_matrices
from einsum.utilities.classes.coo_matrix import Coo_matrix
from timeit import default_timer as timer

if __name__ == "__main__":
    print_results = False
    run_np = False
    run_sql_einsum = False
    run_torch = True

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

    einsum_notation = "ab,bc->ac"

    A = sparse.random((3, 2), density=1.0, idx_dtype=int)
    B = sparse.random((2, 3), density=1.0, idx_dtype=int)

    # A = np.array([[[0.39847798, 0.04342394, 0.25244498, 0.803297],
    #                [0.1446166,  0.36134035, 0.81562155, 0.27616009],
    #                [0.12586615, 0.64966924, 0.78180402, 0.65486967]],

    #               [[0.90505456, 0.39027294, 0.07190914, 0.70894208],
    #                [0.78628425, 0.73966205, 0.90426739, 0.11530047],
    #                [0.86130312, 0.50369939, 0.63523924, 0.85087532]]])

    # B = np.array([[[0.74898691, 0.41647307, 0.0240986],
    #                [0.444104,   0.38949886, 0.46167201],
    #                [0.43550413, 0.9604545,  0.90043466],
    #                [0.07132514, 0.77343843, 0.55865197]],

    #               [[0.65660706, 0.61733519, 0.48561688],
    #                [0.14380024, 0.60082247, 0.1410047],
    #                [0.92153173, 0.81797971, 0.81523549],
    #                [0.35032391, 0.824577,   0.33877537]]])

    # A = sparse.asarray(A)
    # B = sparse.asarray(B)

    sparse_arrays = [A, B]
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
        print(f"Numpy result:\n{numpy_res if run_np else 'None'}")
        print(f"Sparse result:\n{sparse.asnumpy(sparse_res)}\n")
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
