import opt_einsum as oe
import pathlib
import numpy as np
import torch
import sesum.sr as sr
import sqlite3 as sql
from cgreedy import compute_path
from einsum.backends.BMM.bmm_sparse_einsum import sparse_einsum
from einsum.backends.SQL.sql_sparse_einsum import (
    sql_einsum_query, get_matrix_from_sql_response)
from timeit import default_timer as timer


def random_tensor_hypernetwork_benchmark(number_of_repeats=10,
                                         number_of_tensors=40,
                                         regularity=2.5,
                                         max_tensor_order=10,
                                         max_edge_order=5,
                                         number_of_output_indices=5,
                                         min_axis_size=2,
                                         max_axis_size=15,
                                         seed=12345):
    import sparse
    from einsum.utilities.classes.coo_matrix import Coo_tensor
    from einsum_benchmark.generators.random.connected_hypernetwork import connected_hypernetwork as random_tensor_hypernetwork

    torch_time = 0
    sparse_time = 0
    sparse_einsum_time = 0
    correct_results = True

    for _ in range(number_of_repeats):
        eq, shapes, size_dict = random_tensor_hypernetwork(number_of_tensors=number_of_tensors,
                                                           regularity=regularity,
                                                           max_tensor_order=max_tensor_order,
                                                           max_edge_order=max_edge_order,
                                                           number_of_output_indices=number_of_output_indices,
                                                           min_axis_size=min_axis_size,
                                                           max_axis_size=max_axis_size,
                                                           return_size_dict=True,
                                                           seed=seed)

        path, _, _ = compute_path(
            eq,
            *shapes,
            seed=0,
            minimize='size',
            max_repeats=1024,
            max_time=1.0,
            progbar=False,
            is_outer_optimal=False,
            threshold_optimal=12
        )

        torch_tensors = []
        sparse_tensors = []
        dense_tensors = []
        tensor_indices = eq.split("->")[0].split(",")

        for indices in tensor_indices:
            indice_tuple = tuple([size_dict[c] for c in indices])
            sparse_tensor = sparse.random(
                indice_tuple, density=0.01, idx_dtype=int)

            numpy_tensor = sparse.asnumpy(sparse_tensor)
            torch_tensors.append(torch.from_numpy(numpy_tensor))
            sparse_tensors.append(sparse_tensor)
            dense_tensors.append(numpy_tensor)

        # Time opt_einsum with sparse as a backend
        # tic = timer()
        # torch_result = oe.contract(
        #     eq, *torch_tensors, optimize=path, backend='torch')
        # toc = timer()
        # torch_time += toc - tic

        # Time opt_einsum with sparse as a backend
        tic = timer()
        sparse_result = oe.contract(
            eq, *sparse_tensors, optimize=path, backend='sparse')
        toc = timer()
        sparse_time += toc - tic

        # Time sparse_einsum
        tic = timer()
        sparse_einsum_result = sparse_einsum(
            eq, dense_tensors, path=path)
        toc = timer()
        sparse_einsum_time += toc - tic

        if not np.isclose(np.sum(sparse_result), np.sum(sparse_einsum_result)):
            print("Error in calculation: Einsum results not equal!")
            correct_results = False

    torch_time = None if torch_time == 0 else torch_time / number_of_repeats
    sparse_time = sparse_time / number_of_repeats
    sparse_einsum_time = sparse_einsum_time / number_of_repeats

    print("\n------------------------------------------------")
    print("Results of random tensor hypernetwork benchmark:")
    print(f"Torch - average time: {torch_time}s")
    print(f"Sparse - average time: {sparse_time}s")
    print(f"Sparse Einsum - average time: {sparse_einsum_time}s")
    print(f"Results are correct: {'Yes' if correct_results else 'No'}")


def einsum_benchmark_instance_benchmark(instance_name: str,
                                        run_sparse=True,
                                        run_sesum=True,
                                        run_sql_einsum=True,
                                        run_torch=True):
    import einsum_benchmark

    instance = einsum_benchmark.instances[instance_name]
    format_string = instance.format_string
    tensors = instance.tensors
    path_meta = instance.paths.opt_size
    sum_output = instance.result_sum

    # path optimized for minimal intermediate size
    path = path_meta.path
    # print("Size optimized path")
    # print("log10[FLOPS]:", round(path_meta.flops, 2))
    # print("log2[SIZE]:", round(path_meta.size, 2))

    # # path optimized for minimal total flops is stored und the key 1
    # path = instance.paths.opt_flops
    # print("Flops optimized path")
    # print("log10[FLOPS]:", round(flops_log10, 2))
    # print("log2[SIZE]:", round(size_log2, 2))
    if run_sparse:
        try:
            tic = timer()
            result = oe.contract(format_string, *tensors,
                                 optimize=path, backend='sparse')
            toc = timer()
            print("sum[OUTPUT]:", np.sum(result), sum_output)
            print(f"opt_einsum time: {toc - tic}s\n")
        except Exception as e:
            print(e)

    if run_sesum:
        tic = timer()
        result = sr.sesum(format_string, *tensors,
                          path=path, dtype=None,
                          debug=False, safe_convert=False,
                          backend="sparse", semiring=sr.standard)
        toc = timer()
        print("sum[OUTPUT]:", np.sum(np.squeeze(result.data)), sum_output)
        print(f"Sesum time: {toc - tic}s\n")

    if run_sql_einsum:
        file = pathlib.Path(__file__).parent.resolve().joinpath(
            f"{instance_name}_sql_query.sql")

        if file.is_file():
            with open(file, "r") as f:
                query = f.read()
        else:
            with open(file, "w") as f:
                tensor_dict = {}
                tensor_names = []
                for i, arr in enumerate(tensors):
                    tensor_dict["T" + str(i)] = arr
                    tensor_names.append("T" + str(i))

                query = sql_einsum_query(
                    format_string, tensor_names, tensor_dict)
                f.write(query)

        db_connection = sql.connect("SQL_einsum.db")
        db = db_connection.cursor()

        tic = timer()
        result = db.execute(query)
        result = get_matrix_from_sql_response(result.fetchall())
        toc = timer()

        print("sum[OUTPUT]:", result, sum_output)
        print(f"SQL Einsum time: {toc - tic}s")

    if run_torch:
        tic = timer()
        torch_tensors = [torch.from_numpy(i) for i in tensors]
        result = oe.contract(format_string, *torch_tensors,
                             optimize=path, backend='torch')
        toc = timer()
        print("sum[OUTPUT]:", result, sum_output)
        print(f"Torch time: {toc - tic}s")

    tic = timer()
    result = sparse_einsum(format_string, tensors, path=path)
    toc = timer()
    print("sum[OUTPUT]:", np.sum(np.squeeze(result.data)), sum_output)
    print(f"sparse_einsum time: {toc - tic}s\n")


if __name__ == "__main__":
    run_random_tensor_hypernetwork_benchmark = False
    run_einsum_benchmark_instance_benchmark = True

    if run_random_tensor_hypernetwork_benchmark:
        random_tensor_hypernetwork_benchmark()

    if run_einsum_benchmark_instance_benchmark:
        einsum_benchmark_instance_benchmark(
            "mc_2022_079", run_sparse=False, run_torch=False, run_sql_einsum=False)
