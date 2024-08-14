import pathlib
from util import (get_sql_query, get_sql_performance, get_sparse_performance,
                  get_torch_performance, get_sparse_einsum_performance)


def run_instance_experiment(instance_name, iterations=10, run_sparse=True, run_sql_einsum=True, run_torch=False):
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

    # Time opt_einsum sparse backend average execution
    if run_sparse:
        sparse_time, sparse_result = get_sparse_performance(
            iterations, format_string, tensors, path)
        print(f"Sparse time: {sparse_time} iterations per second")

    # Generate SQL einsum query and time average execution
    if run_sql_einsum:
        file = pathlib.Path(__file__).parent.parent.resolve().joinpath(
            "benchmarks", f"{instance_name}_sql_query.sql")

        if file.is_file():
            with open(file, "r") as f:
                query = f.read()
        else:
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
    print(f"Sparse_Einsum time: {sparse_einsum_time} iterations per second")
