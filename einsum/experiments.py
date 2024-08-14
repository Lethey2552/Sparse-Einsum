import opt_einsum as oe
import pathlib
import numpy as np
import torch
import sqlite3 as sql
from cgreedy import compute_path
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, get_matrix_from_sql_response)
from timeit import default_timer as timer


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


def time_n_sparse(n, format_string, tensors, path):
    sparse_time = 0

    try:
        for _ in range(n):
            tic = timer()
            result = oe.contract(format_string, *tensors,
                                 optimize=path, backend='sparse')
            toc = timer()
            sparse_time += toc - tic
    except Exception as e:
        print(e)

    return sparse_time / n


def time_n_sql(n, query):
    db_connection = sql.connect("SQL_einsum.db")
    db = db_connection.cursor()
    sql_time = 0

    try:
        for _ in range(n):
            tic = timer()
            result = db.execute(query)
            result = get_matrix_from_sql_response(result.fetchall())
            toc = timer()
            sql_time += toc - tic
    except Exception as e:
        print(e)

    return sql_time / n


def time_n_torch(n, format_string, tensors, path):
    torch_time = 0

    try:
        for _ in range(n):
            tic = timer()
            torch_tensors = [torch.from_numpy(i) for i in tensors]
            result = oe.contract(format_string, *torch_tensors,
                                 optimize=path, backend='torch')
            toc = timer()
            torch_time += toc - tic
    except Exception as e:
        print(e)

    return torch_time / n


def time_n_sparse_einsum(n, format_string, tensors, path):
    sparse_einsum_time = 0

    try:
        for _ in range(n):
            arrays = tensors[:]
            tic = timer()
            result = sparse_einsum(format_string, arrays,
                                   path=path, show_progress=False, allow_alter_input=True)
            toc = timer()
            sparse_einsum_time += toc - tic
    except Exception as e:
        print(e)

    return sparse_einsum_time / n


if __name__ == "__main__":
    import einsum_benchmark

    NUMBER_OF_RUNS = 10
    RUN_SPARSE = True
    RUN_SQL_EINSUM = False
    RUN_TORCH = False
    INSTANCE_NAME = "mc_2022_087"

    instance = einsum_benchmark.instances[INSTANCE_NAME]
    format_string = instance.format_string
    tensors = instance.tensors
    path_meta = instance.paths.opt_size
    sum_output = instance.result_sum

    # path optimized for minimal intermediate size
    path = path_meta.path
    # print("Size optimized path")
    # print("log10[FLOPS]:", round(path_meta.flops, 2))
    # print("log2[SIZE]:", round(path_meta.size, 2))

    # Time opt_einsum sparse backend average execution
    if RUN_SPARSE:
        sparse_time = time_n_sparse(
            NUMBER_OF_RUNS, format_string, tensors, path)
        print(f"Sparse time: {sparse_time}s\n")

    # Generate SQL einsum query and time average execution
    if RUN_SQL_EINSUM:
        file = pathlib.Path(__file__).parent.resolve().joinpath(
            "benchmarks", f"{INSTANCE_NAME}_sql_query.sql")

        if file.is_file():
            with open(file, "r") as f:
                query = f.read()
        else:
            with open(file, "w") as f:
                query = get_sql_query(format_string, tensors, f)

        sql_time = time_n_sql(NUMBER_OF_RUNS, query)

        print(f"SQL Einsum time: {sql_time}s")

    # Time opt_einsum torch backend average execution
    if RUN_TORCH:
        torch_time = time_n_torch(NUMBER_OF_RUNS, format_string, tensors, path)
        print(f"Torch time: {torch_time}s")

    # Run our sparse_einsum function
    sparse_einsum_time = time_n_sparse_einsum(
        NUMBER_OF_RUNS, format_string, tensors, path)
    print(f"Sparse_Einsum time: {sparse_einsum_time}s\n")
