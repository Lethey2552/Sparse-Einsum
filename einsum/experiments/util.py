import numpy as np
import opt_einsum as oe
import torch
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, sql_einsum_execute)
from timeit import default_timer as timer
import traceback


def delete_values_to_match_density(tensor, target_density):
    current_density = np.count_nonzero(tensor) / tensor.size

    if current_density <= target_density:
        return tensor

    total_elements = tensor.size
    target_non_zero_count = int(np.ceil(target_density * total_elements))

    non_zero_indices = np.flatnonzero(tensor)

    current_non_zero_count = non_zero_indices.size
    elements_to_remove = current_non_zero_count - target_non_zero_count

    # Randomly select indices to zero out
    if elements_to_remove > 0:
        indices_to_delete = np.random.choice(
            non_zero_indices, size=elements_to_remove, replace=False)
        tensor.flat[indices_to_delete] = 0

    return tensor


def get_sql_query(format_string, tensors, file):
    tensor_dict = {}
    tensor_names = []
    for i, arr in enumerate(tensors):
        tensor_dict["T" + str(i)] = arr
        tensor_names.append("T" + str(i))

    query, res_shape = sql_einsum_query(
        format_string, tensor_names, tensor_dict)
    file.write(query)

    return query, res_shape


def get_sparse_performance(n, format_string, tensors, path):
    try:
        sparse_result = oe.contract(format_string, *tensors,
                                    optimize=path, backend='sparse')

        tic = timer()
        for _ in range(n):
            sparse_result = oe.contract(format_string, *tensors,
                                        optimize=path, backend='sparse')
        toc = timer()

        elapsed_time = toc - tic

        # Avoid division by zero
        if elapsed_time <= 0:
            return 0, False

        iterations_per_second = n / elapsed_time

    except Exception as e:
        print(f"Error occurred: {e}")
        return 0, False

    return iterations_per_second, sparse_result


def get_sql_performance(n, query, res_shape, sparse_result=False):

    # Error: Does not work for our nor for Blachers sql_einsum_query function
    try:
        sql_result = sql_einsum_execute(query, res_shape)

        tic = timer()
        for _ in range(n):
            sql_result = sql_einsum_execute(query, res_shape)
        toc = timer()

        if not sparse_result is False:
            if not np.any(sparse_result):
                assert np.isclose(np.sum(sql_result), np.sum(sparse_result))
            else:
                assert np.allclose(sql_result, sparse_result)
    except Exception:
        print(traceback.format_exc())

        return 0

    return 1 / ((toc-tic) / n)


def get_torch_performance(n, format_string, tensors, path, sparse_result):
    try:
        torch_tensors = [torch.from_numpy(i) for i in tensors]
        torch_result = oe.contract(format_string, *torch_tensors,
                                   optimize=path, backend='torch')

        tic = timer()
        for _ in range(n):
            torch_tensors = [torch.from_numpy(i) for i in tensors]
            torch_result = oe.contract(format_string, *torch_tensors,
                                       optimize=path, backend='torch')
        toc = timer()

        if not sparse_result is False:
            assert np.allclose(torch_result, sparse_result)
    except Exception as e:
        print(e)

        return 0

    return 1 / ((toc-tic) / n)


def get_sparse_einsum_performance(n, format_string, tensors, path, sparse_result):
    try:
        arrays = [torch.from_numpy(i) for i in tensors]
        sparse_einsum_result = sparse_einsum(format_string, arrays,
                                             path=path, show_progress=False,
                                             allow_alter_input=True)

        tic = timer()
        for _ in range(n):
            arrays = [torch.from_numpy(i) for i in tensors]
            sparse_einsum_result = sparse_einsum(format_string, arrays,
                                                 path=path, show_progress=False, allow_alter_input=True)
        toc = timer()

        if not sparse_result is False:
            assert np.allclose(sparse_einsum_result, sparse_result)
    except Exception as e:
        print(e)

        return 0

    return 1 / ((toc-tic) / n)
