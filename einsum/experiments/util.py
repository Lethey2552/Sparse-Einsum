import numpy as np
import opt_einsum as oe
import torch
import sqlite3 as sql
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, get_matrix_from_sql_response)
from timeit import default_timer as timer
import traceback


def get_sql_query(format_string, tensors, file):
    tensor_dict = {}
    tensor_names = []
    for i, arr in enumerate(tensors):
        tensor_dict["T" + str(i)] = arr
        tensor_names.append("T" + str(i))

    query = sql_einsum_query(
        format_string, tensor_names, tensor_dict)
    file.write(query)

    return query


def get_sparse_performance(n, format_string, tensors, path):
    try:
        tic = timer()
        for _ in range(n):
            sparse_result = oe.contract(format_string, *tensors,
                                        optimize=path, backend='sparse')
        toc = timer()
    except Exception as e:
        print(e)

    return 1 / ((toc-tic) / n), sparse_result


def get_sql_performance(n, query, sparse_result=False):
    db_connection = sql.connect("SQL_einsum.db")
    db = db_connection.cursor()

    try:
        tic = timer()
        for _ in range(n):
            sql_result = db.execute(query)
            sql_result = get_matrix_from_sql_response(sql_result.fetchall())
        toc = timer()
    except Exception:
        print(traceback.format_exc())

        return 0

    print(sql_result)
    print(sparse_result)

    if not sparse_result is False:
        assert np.allclose(sql_result, sparse_result)

    return 1 / ((toc-tic) / n)


def get_torch_performance(n, format_string, tensors, path, sparse_result):
    try:
        tic = timer()
        for _ in range(n):
            torch_tensors = [torch.from_numpy(i) for i in tensors]
            torch_result = oe.contract(format_string, *torch_tensors,
                                       optimize=path, backend='torch')
        toc = timer()
    except Exception as e:
        print(e)

    if not sparse_result is False:
        assert np.allclose(torch_result, sparse_result)

    return 1 / ((toc-tic) / n)


def get_sparse_einsum_performance(n, format_string, tensors, path, sparse_result):
    try:
        tic = timer()
        for _ in range(n):
            arrays = tensors[:]
            sparse_einsum_result = sparse_einsum(format_string, arrays,
                                                 path=path, show_progress=False, allow_alter_input=True)
        toc = timer()
    except Exception as e:
        print(e)

    if not sparse_result is False:
        assert np.allclose(sparse_einsum_result, sparse_result)

    return 1 / ((toc-tic) / n)
