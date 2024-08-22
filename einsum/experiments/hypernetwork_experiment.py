import numpy as np
import pathlib
import sparse
from cgreedy import compute_path
from einsum_benchmark.generators.random.connected_hypernetwork import connected_hypernetwork as random_tensor_hypernetwork
from util import (get_sql_query, get_sql_performance, get_sparse_performance,
                  get_torch_performance, get_sparse_einsum_performance,
                  delete_values_to_match_density)
from einsum.utilities.classes.coo_matrix import Coo_tensor


def run_hypernetwork_experiment(iterations_per_network=10,
                                iterations=10,
                                run_sparse=True,
                                run_sql_einsum=False,
                                run_torch=False,
                                run_legacy=False,
                                random_hypernetwork_params={
                                    "number_of_tensors": 40,
                                    "regularity": 2.5,
                                    "max_tensor_order": 10,
                                    "max_edge_order": 5,
                                    "number_of_output_indices": 0,
                                    "min_axis_size": 2,
                                    "max_axis_size": 10,
                                    "return_size_dict": True
                                },
                                density=0.001,
                                change="",
                                run_density=True):

    sparse_time = 0
    sql_time = 0
    torch_time = 0
    sparse_einsum_time = 0
    legacy_sparse_einsum_time = 0

    for i in range(iterations):
        print(f"Running iteration {i}/{iterations}...", end="\r")
        print(f"{f'{change}':<15}", end="")

        format_string, shapes, size_dict = random_tensor_hypernetwork(
            seed=i, **random_hypernetwork_params)

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
            numpy_tensor = np.random.rand(*indice_tuple)
            numpy_tensor = delete_values_to_match_density(
                numpy_tensor, density)

            # Make SQL work correctly
            if np.count_nonzero(numpy_tensor) <= 1:
                idx_1 = np.random.choice(
                    numpy_tensor.shape[0], 2, replace=False)

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
            tmp_sparse_time, sparse_result = get_sparse_performance(
                iterations_per_network, format_string, tensors, path)

            sparse_time += tmp_sparse_time

        # Generate SQL einsum query and time average execution
        if run_sql_einsum:
            file = pathlib.Path(__file__).parent.parent.resolve().joinpath(
                "benchmarks", f"sql_query_test.sql")

            with open(file, "w") as f:
                query, res_shape = get_sql_query(format_string, tensors, f)

            sql_time += get_sql_performance(
                iterations_per_network, query, res_shape, sparse_result)

        # Time opt_einsum torch backend average execution
        if run_torch:
            torch_time += get_torch_performance(
                iterations_per_network, format_string, tensors, path, sparse_result)

        # Time opt_einsum torch backend average execution
        if run_legacy:
            legacy_sparse_einsum_time += get_sparse_einsum_performance(
                iterations_per_network, format_string, tensors, path, sparse_result, run_legacy, run_density)

        # Time our sparse_einsum function
        sparse_einsum_time += get_sparse_einsum_performance(
            iterations_per_network, format_string, tensors, path, sparse_result, run_density=run_density)

    output = ""

    if run_sparse:
        output += f"{f'{sparse_time / iterations:.3f} it/s':<15}"

    if run_sql_einsum:
        output += f"{f'{sql_time / iterations:.3f} it/s':<18}"

    if run_torch:
        output += f"{f'{torch_time / iterations:.3f} it/s':<15}"

    output += f"{f'{sparse_einsum_time / iterations:.3f} it/s':<21}"

    if run_legacy:
        output += f"{f'{legacy_sparse_einsum_time / iterations:.3f} it/s'}"

    print(output)


def random_hypernetwork_benchmark_density(iterations_per_network=10,
                                          iterations=10,
                                          run_sparse=True,
                                          run_sql_einsum=False,
                                          run_torch=False,
                                          run_legacy=False,
                                          random_hypernetwork_params={
                                              "number_of_tensors": 40,
                                              "regularity": 2.5,
                                              "max_tensor_order": 10,
                                              "max_edge_order": 5,
                                              "number_of_output_indices": 0,
                                              "min_axis_size": 2,
                                              "max_axis_size": 10,
                                              "return_size_dict": True
                                          },
                                          densities=[0.1, 0.01]):

    sparse_times = []
    sql_times = []
    torch_times = []
    sparse_einsum_times = []
    legacy_sparse_einsum_times = []

    for i in range(len(densities)):
        sparse_times.append(0.0)
        sql_times.append(0.0)
        torch_times.append(0.0)
        sparse_einsum_times.append(0.0)
        legacy_sparse_einsum_times.append(0.0)

    for iter in range(iterations):
        format_string, shapes, size_dict = random_tensor_hypernetwork(
            seed=iter, **random_hypernetwork_params)

        tensors = []
        tensor_indices = format_string.split("->")[0].split(",")

        np.random.seed(seed=0)

        for indices in tensor_indices:
            indice_tuple = tuple([size_dict[c] for c in indices])

            tensors.append(np.random.rand(*indice_tuple))

        for i, density in enumerate(densities):
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

            arrays = []

            for tensor in tensors:
                arrays.append(delete_values_to_match_density(tensor, density))

            sparse_result = False
            # Time opt_einsum sparse backend average execution
            if run_sparse:
                tmp_sparse_time, sparse_result = get_sparse_performance(
                    iterations_per_network, format_string, arrays, path)

                sparse_times[i] += tmp_sparse_time

            # Generate SQL einsum query and time average execution
            if run_sql_einsum:
                file = pathlib.Path(__file__).parent.parent.resolve().joinpath(
                    "benchmarks", f"sql_query_test.sql")

                with open(file, "w") as f:
                    query, res_shape = get_sql_query(format_string, arrays, f)

                sql_times[i] += get_sql_performance(
                    iterations_per_network, query, res_shape, sparse_result)

            # Time opt_einsum torch backend average execution
            if run_torch:
                torch_times[i] += get_torch_performance(
                    iterations_per_network, format_string, arrays, path, sparse_result)

            # Time opt_einsum torch backend average execution
            if run_legacy:
                legacy_sparse_einsum_times[i] += get_sparse_einsum_performance(
                    iterations_per_network, format_string, arrays, path, sparse_result, run_parallel=True, run_density=True)

            # Time our sparse_einsum function
            sparse_einsum_times[i] += get_sparse_einsum_performance(
                iterations_per_network, format_string, arrays, path, sparse_result, run_density=True)

    for i in range(len(densities)):
        output = f"{f'{densities[i]}':<15}"

        if run_sparse:
            output += f"{f'{sparse_times[i] / iterations:.3f} it/s':<15}"

        if run_sql_einsum:
            output += f"{f'{sql_times[i] / iterations:.3f} it/s':<18}"

        if run_torch:
            output += f"{f'{torch_times[i] / iterations:.3f} it/s':<15}"

        output += f"{f'{sparse_einsum_times[i] / iterations:.3f} it/s':<21}"

        if run_legacy:
            output += f"{f'{legacy_sparse_einsum_times[i] / iterations:.3f} it/s'}"

        print(output)
