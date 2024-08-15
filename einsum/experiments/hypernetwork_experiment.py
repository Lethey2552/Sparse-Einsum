import numpy as np
import pathlib
import sparse
from cgreedy import compute_path
from einsum_benchmark.generators.random.connected_hypernetwork import connected_hypernetwork as random_tensor_hypernetwork
from util import (get_sql_query, get_sql_performance, get_sparse_performance,
                  get_torch_performance, get_sparse_einsum_performance)


def run_hypernetwork_experiment(
        iterations=10, run_sparse=True,
        run_sql_einsum=False, run_torch=False,
        random_hypernetwork_params={
            "number_of_tensors": 10,
            "regularity": 2.5,
            "max_tensor_order": 10,
            "max_edge_order": 5,
            "number_of_output_indices": 5,
            "min_axis_size": 2,
            "max_axis_size": 10,
            "return_size_dict": True,
            "seed": 12345
        },
        sparsity=0.001):

    format_string, shapes, size_dict = random_tensor_hypernetwork(
        **random_hypernetwork_params)

    path, _, _ = compute_path(
        format_string,
        *shapes,
        seed=0,
        minimize='size',
        max_repeats=1024,
        max_time=1.0,
        progbar=False,
        is_outer_optimal=False,
        threshold_optimal=12
    )

    tensors = []
    tensor_indices = format_string.split("->")[0].split(",")

    for indices in tensor_indices:
        indice_tuple = tuple([size_dict[c] for c in indices])
        sparse_tensor = sparse.random(
            indice_tuple, density=sparsity, idx_dtype=int)

        numpy_tensor = sparse.asnumpy(sparse_tensor)

        if np.count_nonzero(numpy_tensor) <= 1:
            idx_1 = np.random.choice(numpy_tensor.shape[0], 2, replace=False)

            if len(numpy_tensor.shape) != 1:
                idx_2 = np.random.choice(
                    numpy_tensor.shape[1], 2, replace=False)

                numpy_tensor[idx_1[0], idx_2[0]] = np.random.rand()
                numpy_tensor[idx_1[1], idx_2[1]] = np.random.rand()
            else:
                numpy_tensor[idx_1[0]] = np.random.rand()
                numpy_tensor[idx_1[1]] = np.random.rand()

        tensors.append(numpy_tensor)

    sparse_result = False
    # Time opt_einsum sparse backend average execution
    if run_sparse:
        sparse_time, sparse_result = get_sparse_performance(
            iterations, format_string, tensors, path)
        print(f"Sparse time: {sparse_time} iterations per second")

    # Generate SQL einsum query and time average execution
    if run_sql_einsum:
        file = pathlib.Path(__file__).parent.parent.resolve().joinpath(
            "benchmarks", f"sql_query_test.sql")

        with open(file, "w") as f:
            query = get_sql_query(format_string, tensors, f)

        sql_time = get_sql_performance(iterations, query, sparse_result)

        print(f"SQL Einsum time: {sql_time} iterations per second")

    # Time opt_einsum torch backend average execution
    if run_torch:
        torch_time = get_torch_performance(
            iterations, format_string, tensors, path, sparse_result)
        print(f"Torch time: {torch_time} iterations per second")

    # Time our sparse_einsum function
    sparse_einsum_time = get_sparse_einsum_performance(
        iterations, format_string, tensors, path, sparse_result)
    print(
        f"Sparse_Einsum time: {sparse_einsum_time} iterations per second")
